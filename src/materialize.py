"""
materialize.py — logic thuần cho việc ghi dataset_v2 THẬT từ tập proposed
changes/merges/removals đã validate "valid" (xem src/adjudication.py).
KHÔNG tự quyết định ghi đĩa ở đâu/khi nào -- chỉ nhận input (splits v1,
entities, danh sách op hợp lệ) và trả về data structures đã sẵn sàng
serialize. scripts/apply_adjudication.py (CLI layer) chịu trách nhiệm
atomic write / temp dir / rename.

Nguyên tắc cứng:
- Không sửa list tokens (ký tự) của câu, CHỈ sửa mảng BIO labels.
- Áp toàn bộ "old span" (clear về "O") của MỌI op chạm 1 câu TRƯỚC KHI ghi
  bất kỳ "new span" nào của câu đó -- tránh 1 new span của op A bị hiểu
  nhầm là chồng lên old span của op B (vốn sẽ bị xoá).
- Sau khi ghi hết new span, nếu 2 op (khác nguồn, vd 2 GR-11 merge riêng
  biệt trong cùng câu) vô tình ghi đè lên nhau -> raise ValueError ngay
  (cross-op collision), KHÔNG âm thầm ghi đè -- validate_and_build_merge/
  check_no_overlap_after_edit ở adjudication.py chỉ kiểm tra từng dòng
  workbook riêng lẻ với entity index GỐC, không bắt được xung đột giữa 2
  op khác nhau trong cùng 1 câu.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.audit_utils import ENTITY_TYPES, bio_to_entities

LABEL_SCHEME = "BIO"
OFFSET_CONVENTION = "inclusive"


# ── Op grouping ──────────────────────────────────────────────────────────
@dataclass
class SentenceOp:
    """1 op (change/merge/removal) đã chuẩn hoá về dạng chung: xoá 0+ old
    range, ghi 0/1 new range."""
    change_id: int
    kind: str  # "change" | "merge" | "removal"
    old_ranges: list  # [(start, end), ...] -- range se bi clear ve "O"
    new_range: Optional[tuple]  # (start, end, label) hoặc None (removal)
    rule_id: str
    reviewer_decision: str
    source_workbook_row: str
    source_obj: object  # ProposedChange | ProposedMerge | ProposedRemoval goc, de format sources


def _format_op_sources(op: "SentenceOp") -> str:
    """'LABEL[start-end]; LABEL[start-end]' -- dung chung 1 format voi
    original_entities cua GR-11 workbook de de doi chieu."""
    obj = op.source_obj
    if op.kind == "merge":
        return "; ".join(f"{lbl}[{s}-{e}]" for s, e, lbl, _surface in obj.source_spans)
    # change hoac removal: 1 old span/label duy nhat
    s, e = op.old_ranges[0]
    return f"{obj.old_label}[{s}-{e}]"


def _source_workbook_name(source_workbook_row: str) -> str:
    return source_workbook_row.split(":", 1)[0] if source_workbook_row else ""


def _ops_from_valid(valid_changes: list, valid_merges: list, valid_removals: list) -> dict:
    """-> {(split, sample_id): [SentenceOp, ...]}"""
    by_sentence = defaultdict(list)
    for c in valid_changes:
        by_sentence[(c.split, c.sample_id)].append(SentenceOp(
            change_id=c.change_id, kind="change", old_ranges=[tuple(c.old_span)],
            new_range=(c.new_span[0], c.new_span[1], c.new_label),
            rule_id=c.rule_id, reviewer_decision=c.reviewer_decision,
            source_workbook_row=c.source_workbook_row, source_obj=c,
        ))
    for m in valid_merges:
        by_sentence[(m.split, m.sample_id)].append(SentenceOp(
            change_id=m.change_id, kind="merge",
            old_ranges=[(s[0], s[1]) for s in m.source_spans],
            new_range=(m.resulting_span[0], m.resulting_span[1], m.resulting_label),
            rule_id=m.rule_id, reviewer_decision="MERGE_ENTITY",
            source_workbook_row=m.source_workbook_row, source_obj=m,
        ))
    for r in valid_removals:
        by_sentence[(r.split, r.sample_id)].append(SentenceOp(
            change_id=r.change_id, kind="removal", old_ranges=[tuple(r.old_span)],
            new_range=None, rule_id=r.rule_id, reviewer_decision="REMOVE_ENTITY",
            source_workbook_row=r.source_workbook_row, source_obj=r,
        ))
    return by_sentence


def _labels_for_range(start: int, end: int, label: str) -> list:
    if start == end:
        return [f"B-{label}"]
    return [f"B-{label}"] + [f"I-{label}"] * (end - start)


def _dedupe_redundant_ops(ops: list) -> tuple:
    """
    Trong 1 câu, 2+ op có THỂ trùng đích (new_range giống hệt nhau) nếu
    cùng 1 correction thật được review 2 lần độc lập -- thấy thật trên
    corpus DVSKTT: (a) cùng entity_id được duyệt qua CẢ boundary_split_
    fill_in.xlsx LẪN gr11_date_formula_candidates.xlsx (2 workbook khác
    nhau, cùng kết luận); (b) 2 candidate GR-11 khác candidate_span nhưng
    1 cái là tập con phạm vi của cái kia, cùng final_span/label (vd
    candidate 148-152 chỉ phủ 莫大正 nằm trong candidate 146-152 phủ cả
    元和元年+莫大正). Áp CẢ hai sẽ ghi đè cùng span 2 lần -> cross-op
    collision giả (không phải mâu thuẫn thật, chỉ là dư thừa).

    Quy tắc: nếu op A có old_ranges (hợp các ký tự sẽ bị xoá) là SUPERSET
    của op B (cùng new_range) -> B dư thừa hoàn toàn so với A (áp A đã đủ
    tạo đúng kết quả cuối), bỏ B, ghi log. Nếu old_ranges của nhóm ops
    cùng new_range KHÔNG có quan hệ superset (rời nhau hoặc giao nhau một
    phần) -> đây là mâu thuẫn THẬT hoặc không đủ rõ để tự quyết, GIỮ
    NGUYÊN cả nhóm để Phase 2 tự phát hiện collision và báo lỗi tường
    minh (không tự ý chọn bên nào đúng).
    """
    by_target = defaultdict(list)
    no_target = []
    for op in ops:
        if op.new_range is None:
            no_target.append(op)
        else:
            by_target[op.new_range].append(op)

    keep = []
    drop_records = []
    for new_range, group in by_target.items():
        if len(group) == 1:
            keep.append(group[0])
            continue
        op_covs = []
        for op in group:
            cov = set()
            for s, e in op.old_ranges:
                cov |= set(range(s, e + 1))
            op_covs.append((op, cov))
        op_covs.sort(key=lambda t: -len(t[1]))
        richest_op, richest_cov = op_covs[0]
        if all(cov.issubset(richest_cov) for _, cov in op_covs[1:]):
            keep.append(richest_op)
            fs, fe, flabel = new_range
            for op, cov in op_covs[1:]:
                reason = ("cross_workbook_duplicate_identical_old_ranges" if cov == richest_cov
                          else "cross_workbook_subset_subsumption")
                drop_records.append({
                    "dropped_operation_id": op.change_id,
                    "kept_operation_id": richest_op.change_id,
                    "dropped_action": op.reviewer_decision,
                    "kept_action": richest_op.reviewer_decision,
                    "dropped_sources": _format_op_sources(op),
                    "kept_sources": _format_op_sources(richest_op),
                    "final_target_span": f"{fs}-{fe}",
                    "final_target_label": flabel,
                    "dedup_reason": reason,
                    "source_workbook": _source_workbook_name(op.source_workbook_row),
                    "rule_id": op.rule_id,
                    "dropped_source_workbook_row": op.source_workbook_row,
                    "kept_source_workbook_row": richest_op.source_workbook_row,
                })
        else:
            keep.extend(group)  # mau thuan that -- giu nguyen, de Phase 2 bao loi
    return keep + no_target, drop_records


def apply_ops_to_splits(
    splits_v1: dict, valid_changes: list, valid_merges: list, valid_removals: list,
) -> tuple:
    """
    splits_v1: {"train": [(tokens, labels), ...], ...} (đọc từ read_conll,
    KHÔNG bị sửa in-place -- hàm này luôn deepcopy).

    Trả (new_splits, applied_entities, apply_errors, dedup_notes, dedup_records):
    - new_splits: cùng schema splits_v1 nhưng labels đã áp toàn bộ op valid.
    - applied_entities: {(split, sample_id): [{"start","end","label",
      "surface","provenance": {"change_id","rule_id","reviewer_decision"}
      hoặc None nếu entity không bị đụng tới}]} -- suy lại từ labels cuối
      cùng qua bio_to_entities (nguồn sự thật duy nhất, tránh lệch giữa
      "entity list mình tưởng" và "label array thực sự ghi ra").
    - apply_errors: list[str] -- rỗng nếu mọi câu áp op thành công không
      xung đột. Có lỗi thì new_splits vẫn được trả về (caller quyết định
      abort, KHÔNG tự ý ghi đĩa khi list này khác rỗng).
    - dedup_notes: list[str] -- mô tả người-đọc-được của op bị bỏ vì
      REDUNDANT (xem _dedupe_redundant_ops), KHÔNG phải lỗi.
    - dedup_records: list[dict] -- cùng thông tin, dạng structured (cột
      dropped_operation_id/kept_operation_id/sample_id/split/dropped_action/
      kept_action/dropped_sources/kept_sources/final_target_span/
      final_target_label/dedup_reason/source_workbook/rule_id) để ghi
      changelog/redundant_ops_dropped.csv.
    """
    new_splits = {split: [(list(toks), list(labs)) for toks, labs in sents]
                  for split, sents in splits_v1.items()}
    ops_by_sentence = _ops_from_valid(valid_changes, valid_merges, valid_removals)
    applied_entities = {}
    apply_errors = []
    dedup_notes = []
    dedup_records = []

    touched = set()
    for split, sents in new_splits.items():
        for idx in range(len(sents)):
            touched.add((split, idx))
    # chỉ cần xử lý câu có op, nhưng vẫn build applied_entities cho MỌI câu
    # (kể cả không có op) để JSONL đủ toàn bộ corpus.

    for (split, sample_id), raw_ops in ops_by_sentence.items():
        if split not in new_splits or not (0 <= sample_id < len(new_splits[split])):
            apply_errors.append(f"[{split}:{sample_id}] sample_id ngoai pham vi split khi apply")
            continue
        ops, drop_records_here = _dedupe_redundant_ops(raw_ops)
        for rec in drop_records_here:
            rec = dict(rec, split=split, sample_id=sample_id)
            dedup_records.append(rec)
            dedup_notes.append(
                f"[{split}:{sample_id}] change_id={rec['dropped_operation_id']} "
                f"action={rec['dropped_action']} wbrow={rec['dropped_source_workbook_row']} "
                f"la REDUNDANT so voi change_id={rec['kept_operation_id']} "
                f"action={rec['kept_action']} wbrow={rec['kept_source_workbook_row']} "
                f"(target={rec['final_target_span']}/{rec['final_target_label']}, "
                f"reason={rec['dedup_reason']}) -- KHONG apply."
            )
        tokens, labels = new_splits[split][sample_id]
        text_len = len(tokens)

        # Phase 1: clear toan bo old range cua MOI op truoc
        for op in ops:
            for s, e in op.old_ranges:
                if not (0 <= s <= e < text_len):
                    apply_errors.append(
                        f"[{split}:{sample_id}] op change_id={op.change_id} "
                        f"old_range {(s, e)} ngoai pham vi text_len={text_len}"
                    )
                    continue
                for i in range(s, e + 1):
                    labels[i] = "O"

        # Phase 2: ghi new range, phat hien collision giua cac op KHAC nhau
        new_range_map = {}  # (start,end,label) -> op (cho provenance lookup)
        for op in ops:
            if op.new_range is None:
                continue
            s, e, lbl = op.new_range
            if not (0 <= s <= e < text_len):
                apply_errors.append(
                    f"[{split}:{sample_id}] op change_id={op.change_id} "
                    f"new_range {(s, e)} ngoai pham vi text_len={text_len}"
                )
                continue
            if any(labels[i] != "O" for i in range(s, e + 1)):
                apply_errors.append(
                    f"[{split}:{sample_id}] op change_id={op.change_id} rule={op.rule_id} "
                    f"new_range {(s, e)}/{lbl} DE LEN mot new_range khac trong cung cau "
                    f"(cross-op collision, khong duoc phat hien luc dry-run validate tung dong)"
                )
                continue
            for i, lab in enumerate(_labels_for_range(s, e, lbl)):
                labels[s + i] = lab
            new_range_map[(s, e, lbl)] = op

        new_splits[split][sample_id] = (tokens, labels)
        _record_applied_entities(applied_entities, split, sample_id, tokens, labels, new_range_map)

    # cac cau khong co op nao -- van build applied_entities (provenance=None)
    for split, sents in new_splits.items():
        for idx, (tokens, labels) in enumerate(sents):
            if (split, idx) not in applied_entities:
                _record_applied_entities(applied_entities, split, idx, tokens, labels, {})

    return new_splits, applied_entities, apply_errors, dedup_notes, dedup_records


def _record_applied_entities(applied_entities, split, sample_id, tokens, labels, new_range_map):
    ents = bio_to_entities(tokens, labels)
    text = "".join(tokens)
    recs = []
    for e in ents:
        s, en, lbl = e["start"], e["end"], e["type"]
        op = new_range_map.get((s, en, lbl))
        recs.append({
            "start": s, "end": en, "label": lbl, "surface": text[s:en + 1],
            "provenance": None if op is None else {
                "change_id": op.change_id, "rule_id": op.rule_id,
                "reviewer_decision": op.reviewer_decision,
                "source_workbook_row": op.source_workbook_row,
            },
        })
    applied_entities[(split, sample_id)] = recs


# ── Quarantine split ────────────────────────────────────────────────────
def quarantine_sample_ids_by_split(all_quarantined: list) -> dict:
    """all_quarantined: list[SkippedDecision] (reviewer_decision ==
    SKIP_UNANNOTATED_SAMPLE) -> {"train": {12, 45, ...}, "dev": {...}, ...}"""
    out = defaultdict(set)
    for q in all_quarantined:
        if q.split is None or q.sample_id is None:
            continue
        try:
            out[q.split].add(int(q.sample_id))
        except (TypeError, ValueError):
            continue
    return dict(out)


def build_clean_splits(new_splits: dict, quarantine_ids: dict) -> tuple:
    """
    -> (clean_splits, index_map) -- index_map[split] = {old_idx: new_idx}
    cho MỌI sample GIỮ LẠI (không có entry cho sample bị quarantine), để
    truy vết vị trí dòng mới trong file .txt/.jsonl sau khi lọc.
    """
    clean_splits = {}
    index_map = {}
    for split, sents in new_splits.items():
        qids = quarantine_ids.get(split, set())
        kept = []
        idx_map = {}
        new_idx = 0
        for old_idx, sent in enumerate(sents):
            if old_idx in qids:
                continue
            kept.append(sent)
            idx_map[old_idx] = new_idx
            new_idx += 1
        clean_splits[split] = kept
        index_map[split] = idx_map
    return clean_splits, index_map


def build_quarantine_records(splits_v1: dict, quarantine_ids: dict, all_quarantined: list) -> dict:
    """Trả {"train": [record, ...], ...} -- record giữ NGUYÊN tokens/labels
    GỐC v1 (không áp correction nào) cho mỗi sample bị quarantine, kèm
    danh sách reason/candidate/reviewer_notes gộp theo sample_id (1 sample
    có thể có nhiều dòng GR-11 SKIP_UNANNOTATED_SAMPLE)."""
    reasons_by_sample = defaultdict(list)
    for q in all_quarantined:
        if q.split is None or q.sample_id is None:
            continue
        try:
            sid = int(q.sample_id)
        except (TypeError, ValueError):
            continue
        reasons_by_sample[(q.split, sid)].append({
            "source_workbook_row": q.source_workbook_row, "surface": q.surface,
            "reviewer_decision": q.reviewer_decision, "reason": q.reason,
        })

    out = {}
    for split, qids in quarantine_ids.items():
        sents = splits_v1.get(split, [])
        recs = []
        for sid in sorted(qids):
            if not (0 <= sid < len(sents)):
                continue
            tokens, labels = sents[sid]
            text = "".join(tokens)
            ents = bio_to_entities(tokens, labels)
            recs.append({
                "split": split, "sample_id": sid, "text_raw": text,
                "tokens": tokens, "bio_labels": labels,
                "entities": [
                    {"start": e["start"], "end": e["end"], "label": e["type"],
                     "surface": text[e["start"]:e["end"] + 1]}
                    for e in ents
                ],
                "quarantine_candidates": reasons_by_sample.get((split, sid), []),
                "suggested_action": "COMPLETE_FULL_ANNOTATION_BEFORE_REUSE",
            })
        out[split] = recs
    return out


# ── BIO validation ───────────────────────────────────────────────────────
def validate_bio_labels(tokens: list, labels: list) -> list:
    """Kiểm tra sequence BIO hợp lệ (không I-X mồ côi, độ dài khớp, nhãn
    thuộc ENTITY_TYPES). Trả list lỗi (rỗng nếu OK)."""
    errors = []
    if len(tokens) != len(labels):
        errors.append(f"token/label length mismatch ({len(tokens)} vs {len(labels)})")
        return errors
    prev = "O"
    for i, lab in enumerate(labels):
        if lab == "O":
            prev = lab
            continue
        if "-" not in lab:
            errors.append(f"pos {i}: invalid label format {lab!r}")
            prev = lab
            continue
        prefix, etype = lab.split("-", 1)
        if prefix not in ("B", "I"):
            errors.append(f"pos {i}: invalid BIO prefix {lab!r}")
        if etype not in ENTITY_TYPES:
            errors.append(f"pos {i}: unknown entity type {etype!r}")
        if prefix == "I" and prev not in (f"B-{etype}", f"I-{etype}"):
            errors.append(f"pos {i}: orphan I-{etype} (prev={prev!r})")
        prev = lab
    return errors


def check_no_flat_overlap(entities: list) -> list:
    """entities: list[{"start","end",...}] CÙNG 1 câu, đã sort hay chưa
    không quan trọng. Trả list mô tả overlap nếu có (rỗng nếu flat OK)."""
    errors = []
    sorted_ents = sorted(entities, key=lambda e: e["start"])
    for a, b in zip(sorted_ents, sorted_ents[1:]):
        if a["end"] >= b["start"]:
            errors.append(f"overlap {a} vs {b}")
    return errors


# ── Serialization ────────────────────────────────────────────────────────
def write_conll(path: str, sentences: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for tokens, labels in sentences:
            for tok, lab in zip(tokens, labels):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


def build_jsonl_records(sentences: list, split: str, applied_entities: dict,
                         source_map: dict, dataset_version: str) -> list:
    records = []
    for idx, (tokens, labels) in enumerate(sentences):
        text = "".join(tokens)
        recs = applied_entities.get((split, idx), [])
        entities = []
        for r in recs:
            prov = r["provenance"] or {}
            entities.append({
                "entity_id": f"{split}-{idx}-{r['start']}-{r['end']}",
                "start": r["start"], "end": r["end"],
                "offset_convention": OFFSET_CONVENTION,
                "surface": r["surface"], "label": r["label"],
                "provenance": {
                    "dataset_version": dataset_version,
                    "change_id": prov.get("change_id"),
                    "rule_id": prov.get("rule_id"),
                },
            })
        records.append({
            "sample_id": idx, "split": split,
            "document_id": source_map.get((split, idx)),
            "text_raw": text, "text_normalized": text,
            "normalization_applied": False,
            "tokens": tokens, "bio_labels": labels,
            "entities": entities,
        })
    return records


def write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_dir_checksums(root: str) -> dict:
    """md5 cho moi file thuong duoi root (duong dan tuong doi voi root, /
    separator), bo qua checksums.json chinh no neu ton tai."""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "checksums.json":
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            h = hashlib.md5()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[rel] = h.hexdigest()
    return dict(sorted(out.items()))


def entity_stats(sentences_by_split: dict) -> dict:
    """{"split": {"num_sentences": N, "num_entities": M,
    "label_distribution": {"PER": n, ...}}}"""
    out = {}
    for split, sents in sentences_by_split.items():
        n_ent = 0
        dist = defaultdict(int)
        for tokens, labels in sents:
            for e in bio_to_entities(tokens, labels):
                n_ent += 1
                dist[e["type"]] += 1
        out[split] = {
            "num_sentences": len(sents), "num_entities": n_ent,
            "label_distribution": dict(sorted(dist.items())),
        }
    return out


def _count_by_split(split_values: list) -> dict:
    out = defaultdict(int)
    for s in split_values:
        out[s] += 1
    return dict(out)


def build_critical_samples_report(splits_v1: dict, applied_entities: dict, targets: list) -> list:
    """targets: [(split, sample_id), ...]. Trả list record {split, sample_id,
    raw_text, v1_entities, v2_full_entities} de nhung sample quan trong
    (GR-11, transaction, merge, quarantine, GR-11b, ADD_ENTITY) luon co
    trong report cuoi cung, khong can tra cuu thu cong."""
    out = []
    for split, sid in targets:
        sents = splits_v1.get(split, [])
        if not (0 <= sid < len(sents)):
            out.append({"split": split, "sample_id": sid, "error": "sample_id_out_of_range"})
            continue
        tokens, labels = sents[sid]
        text = "".join(tokens)
        v1_entities = [
            {"label": e["type"], "start": e["start"], "end": e["end"],
             "surface": text[e["start"]:e["end"] + 1]}
            for e in bio_to_entities(tokens, labels)
        ]
        v2_entities = [
            {"label": e["label"], "start": e["start"], "end": e["end"], "surface": e["surface"],
             "provenance": e["provenance"]}
            for e in applied_entities.get((split, sid), [])
        ]
        out.append({
            "split": split, "sample_id": sid, "raw_text": text,
            "v1_entities": v1_entities, "v2_full_entities": v2_entities,
        })
    return out


def git_commit_hash(repo_dir: str) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
            text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def environment_info() -> str:
    return (
        f"python={sys.version.split()[0]}\n"
        f"platform={platform.platform()}\n"
        f"executable={sys.executable}\n"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Content generators (guideline / dataset card / label schema) ─────────
def label_schema_dict() -> dict:
    labels = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
    return {
        "entity_types": list(ENTITY_TYPES), "scheme": LABEL_SCHEME,
        "labels": labels, "offset_convention": OFFSET_CONVENTION,
        "flat_ner_no_nesting": True,
    }


def normalization_config_yaml() -> str:
    return (
        "# normalization_config.yaml\n"
        "normalization_applied: false\n"
        "pua_normalization_applied: false\n"
        "note: >\n"
        "  Chua co normalization pipeline versioned/validated trong lan\n"
        "  materialize nay. text_normalized == text_raw trong moi JSONL\n"
        "  record. KHONG duoc suy dien la da normalize.\n"
    )


def guideline_v2_yaml(source_yaml_path: str, applied_gr11: bool) -> str:
    src = ""
    if source_yaml_path and os.path.exists(source_yaml_path):
        with open(source_yaml_path, encoding="utf-8") as f:
            src = f.read()
    header = (
        "status: frozen_for_dataset_v2\n"
        f"frozen_at_utc: \"{utc_now_iso()}\"\n"
        "inherited_source: guideline_v1\n"
        "schema: [PER, LOC, ORG, TITLE, DTM]\n"
        "schema_note: \"EVENT chua duoc them vao schema v2 (out-of-schema, deferred)\"\n"
        "flat_ner_no_nesting: true\n"
        "offset_convention: inclusive\n"
        "pua_normalization_status: \"KHONG duoc apply trong dataset_v2 (chua co pipeline versioned/validated)\"\n"
        f"applied_rules: [{'GR-11' if applied_gr11 else ''}]\n"
        "deferred_rules: [GR-11b]\n"
        "---\n"
        "# Noi dung ke thua tu guideline_v2.0_draft.yaml (v1 -> draft) ben duoi:\n"
    )
    return header + src


def guideline_v2_md(applied_counts: dict) -> str:
    return (
        "# Guideline v2.0 (frozen for dataset_v2)\n\n"
        f"- Frozen at (UTC): {utc_now_iso()}\n"
        "- Inherited source: guideline v1\n"
        "- Schema (KHONG doi so voi v1): PER, LOC, ORG, TITLE, DTM. EVENT "
        "chua duoc them (out-of-schema, deferred).\n"
        "- Flat NER, KHONG nesting entity.\n"
        "- Offset convention: inclusive (start,end đều nằm trong entity).\n"
        "- PUA normalization: CHƯA áp dụng (không có pipeline versioned/"
        "validated ở lần materialize này) -- text_normalized == text_raw.\n"
        "- Rule đã áp trong dataset_v2: **GR-11** (whole-span DTM cho công "
        f"thức ngày tháng hoàn chỉnh) -- {applied_counts.get('gr11_changes', 0)} "
        f"CORRECT_BOUNDARY đơn-nguồn + {applied_counts.get('gr11_merges', 0)} "
        "MERGE_ENTITY đa-nguồn.\n"
        "- Rule deferred: **GR-11b** (dynasty_era_candidates.xlsx, "
        "scan_dynasty_era_candidates.py, dựa trên policy GR-11 CŨ đã bị "
        "đảo ngược -- cần re-review dưới policy whole-span mới trước khi "
        "dùng, KHÔNG dùng lại kết luận cũ).\n"
        "- Xem docs/guideline_v2.0_draft.yaml (đính kèm nguyên văn trong "
        "guideline_v2.0.yaml) cho toàn bộ rule khác GR-01..GR-13+.\n"
    )


def render_dataset_card(
    variant: str, split_stats: dict, applied_counts: dict,
    quarantine_decision_rows: int, quarantine_unique_records: int,
) -> str:
    lines = [
        f"# DVSKTT NER dataset_v2 ({variant})",
        "",
        "## Nguồn",
        "- Corpus gốc: Đại Việt Sử Ký Toàn Thư (DVSKTT), dataset_v1 "
        "(gold NER sau gold-review v2.1).",
        "- Mục tiêu: NER Hán văn cổ Việt Nam (PER/LOC/ORG/TITLE/DTM).",
        "",
        "## Label schema",
        f"- {', '.join(ENTITY_TYPES)} (scheme {LABEL_SCHEME}, offset inclusive).",
        "",
        "## Split statistics (tính lại từ output thực tế, không hardcode)",
    ]
    for split, st in split_stats.items():
        lines.append(f"- **{split}**: {st['num_sentences']} câu, {st['num_entities']} entity "
                      f"({', '.join(f'{k}={v}' for k, v in st['label_distribution'].items())})")
    lines += [
        "",
        "## Thay đổi áp dụng so với dataset_v1",
        f"- Review-level (dry-run, trước materializer dedup): "
        f"{applied_counts.get('n_changes_reviewlevel', 0)} direct edit, "
        f"{applied_counts.get('n_merges_reviewlevel', 0)} merge.",
        f"- **Actually written** (sau materializer redundancy dedup — xem "
        f"changelog/redundant_ops_dropped.csv): "
        f"{applied_counts.get('n_changes_written', 0)} direct edit "
        f"(CORRECT_LABEL + GR-11 CORRECT_BOUNDARY đơn-nguồn), "
        f"{applied_counts.get('n_merges_written', 0)} merge "
        f"(GR-11 MERGE_ENTITY đa-nguồn + GR-12).",
        f"- {applied_counts.get('n_removals', 0)} removal (transaction 故都/LOC + CORRECT_BOUNDARY 御史->都御史, atomic).",
        f"- Quarantine (SKIP_UNANNOTATED_SAMPLE): **{quarantine_decision_rows} decision rows** "
        f"(1 sample có thể có nhiều candidate date-formula chưa gắn nhãn) tương ứng "
        f"**{quarantine_unique_records} unique sample_id** thực sự bị loại khỏi v2_clean "
        f"-- KHÔNG gọi {quarantine_decision_rows} là số 'sample'.",
        "- Policy: GR-11 whole-span DTM cho công thức ngày tháng hoàn chỉnh.",
        "",
        "## Vấn đề chưa xử lý / deferred",
        "- GR-11b (dynasty_era_candidates.xlsx, 524 candidate, policy GR-11 CŨ) -- deferred, cần re-review.",
        "- NEEDS_DOMAIN_EXPERT, REJECT_CANDIDATE, GUIDELINE_AMBIGUITY -- chưa auto-apply, xem changelog/unresolved_decisions.csv.",
        "- ADD_ENTITY (2 dòng GR-11) -- chưa được xác nhận an toàn để tự thêm entity mới, chưa áp.",
        "- EVENT (out-of-schema) -- chưa thêm vào schema v2.",
        "",
        "## Cảnh báo quan trọng",
        "- **v2_clean có train/dev/test composition KHÁC v1** (đã bỏ sample "
        "quarantine) -- KHÔNG được so sánh trực tiếp F1 trên v2_clean test "
        "với baseline paper-v1 (test F1 = 0.672), vì test set không còn "
        "giống hệt nhau.",
        "",
        "## Intended use",
        "- **v2_full**: provenance/audit đầy đủ, giữ nguyên mọi sample "
        "(kể cả quarantine) với label đã sửa nếu có -- KHÔNG dùng để so "
        "sánh F1 trực tiếp vì vẫn còn sample thiếu annotation nền.",
        "- **v2_clean**: dùng cho supervised NER training/eval, SAU KHI đã "
        "xem xét ảnh hưởng thay đổi split composition.",
        "- **quarantine_v2**: CHỈ dùng để hoàn thiện annotation "
        "(COMPLETE_FULL_ANNOTATION_BEFORE_REUSE), KHÔNG dùng train/eval.",
        "",
        "## License / nguồn dữ liệu",
        "- (placeholder) Theo license/điều khoản của corpus DVSKTT nguồn "
        "và annotation nội bộ -- cập nhật khi có quyết định chính thức.",
    ]
    return "\n".join(lines) + "\n"


# ── Post-apply effect verification (section D.2) ─────────────────────────
def verify_applied_effects(applied_entities: dict, valid_changes: list,
                            valid_merges: list, valid_removals: list) -> list:
    """Kiểm tra hiệu ứng thực tế của từng op trên applied_entities (suy từ
    label array cuối cùng, xem apply_ops_to_splits) khớp ý định:
    - removal: old (span,label) không còn tồn tại.
    - change: old (span,label) không còn (trừ khi trùng new), new tồn tại
      đúng 1 lần.
    - merge: MỌI source (span,label) không còn (trừ khi trùng resulting),
      resulting tồn tại đúng 1 lần."""
    errors = []

    def _ents_at(split, sample_id):
        return applied_entities.get((split, sample_id), [])

    def _count(ents, s, e, lbl):
        return sum(1 for x in ents if x["start"] == s and x["end"] == e and x["label"] == lbl)

    for r in valid_removals:
        ents = _ents_at(r.split, r.sample_id)
        s, e = r.old_span
        if _count(ents, s, e, r.old_label) != 0:
            errors.append(f"removal change_id={r.change_id}: old entity {r.old_label}[{s}-{e}] van con sau apply")

    for c in valid_changes:
        ents = _ents_at(c.split, c.sample_id)
        os_, oe = c.old_span
        ns, ne = c.new_span
        if (os_, oe, c.old_label) != (ns, ne, c.new_label) and _count(ents, os_, oe, c.old_label) != 0:
            errors.append(f"change change_id={c.change_id}: old entity {c.old_label}[{os_}-{oe}] van con sau apply")
        if _count(ents, ns, ne, c.new_label) != 1:
            errors.append(f"change change_id={c.change_id}: new entity {c.new_label}[{ns}-{ne}] khong ton tai dung 1 lan")

    for m in valid_merges:
        ents = _ents_at(m.split, m.sample_id)
        rs, re = m.resulting_span
        for s, e, lbl, _surface in m.source_spans:
            if (s, e, lbl) != (rs, re, m.resulting_label) and _count(ents, s, e, lbl) != 0:
                errors.append(f"merge change_id={m.change_id}: source {lbl}[{s}-{e}] van con sau apply")
        if _count(ents, rs, re, m.resulting_label) != 1:
            errors.append(f"merge change_id={m.change_id}: resulting entity {m.resulting_label}[{rs}-{re}] khong ton tai dung 1 lan")

    return errors


def verify_no_duplicate_sample_ids(sentences_by_split: dict) -> list:
    """Kiem tra khong co sample_id trung nhau TRONG 1 split o output cuoi
    cung -- suy tu do dai list lien tuc (0..N-1) nen ve nguyen tac khong
    the trung, nhung van assert tuong minh o day thay vi chi tin cau truc
    ngam, phong truong hop code sau nay doi sang dict/list khong lien tuc."""
    errors = []
    for split, sents in sentences_by_split.items():
        ids = list(range(len(sents)))
        if len(ids) != len(set(ids)):
            errors.append(f"[{split}] duplicate sample_id phat hien trong output")
    return errors


def verify_quarantine_partition(new_splits: dict, clean_splits: dict, quarantine_ids: dict) -> list:
    """v2_full PHAI chua toan bo sample quarantine (khong bi loai); v2_clean
    PHAI loai CHINH XAC dung nhung sample_id trong quarantine_ids, khong
    thieu khong thua."""
    errors = []
    for split in new_splits:
        full_n = len(new_splits[split])
        qids = quarantine_ids.get(split, set())
        if any(q >= full_n for q in qids):
            errors.append(f"[{split}] quarantine sample_id vuot qua so cau v2_full")
        clean_n = len(clean_splits.get(split, []))
        expected_clean_n = full_n - len(qids)
        if clean_n != expected_clean_n:
            errors.append(
                f"[{split}] v2_clean co {clean_n} cau, ky vong {expected_clean_n} "
                f"(v2_full={full_n} - quarantine={len(qids)})"
            )
    return errors


def verify_add_entity_never_applied(valid_changes: list, valid_merges: list, valid_removals: list) -> list:
    """Phong ve chieu sau: dam bao KHONG co op nao trong tap da apply mang
    dau vet ADD_ENTITY (2 dong GR-11 chua duoc xac nhan an toan de tu them
    entity) -- process_gr11_date_formula_candidates() da chan tu logic,
    day la lop kiem tra doc lap them o thoi diem materialize."""
    errors = []
    for c in valid_changes:
        if c.reviewer_decision == "ADD_ENTITY":
            errors.append(f"change_id={c.change_id}: reviewer_decision=ADD_ENTITY nhung DA duoc apply")
    for m in valid_merges:
        # ProposedMerge khong luu reviewer_decision -- kiem tra qua rule_id
        # (ADD_ENTITY khong bao gio sinh ProposedMerge trong code hien tai,
        # check nay chi de bat regression neu logic thay doi sau nay).
        if str(m.rule_id).upper() == "ADD_ENTITY":
            errors.append(f"merge change_id={m.change_id}: rule_id=ADD_ENTITY nhung DA duoc apply")
    for r in valid_removals:
        if r.remove_reason and "ADD_ENTITY" in str(r.remove_reason).upper():
            errors.append(f"removal change_id={r.change_id}: remove_reason nhac ADD_ENTITY bat thuong")
    return errors


def verify_source_files_unchanged(source_paths: dict, checksums_before: dict) -> list:
    """source_paths: {"train": path, "dev": path, "test": path}. So checksum
    hien tai cua CHINH FILE v1 tren dia voi checksum da tinh truoc khi
    materialize -- script nay khong bao gio mo cac path nay o mode ghi,
    nhung assert tuong minh o day thay vi chi tin vao "khong ai goi open(w)"."""
    errors = []
    for split, path in source_paths.items():
        if not path or not os.path.exists(path):
            continue
        cur = hashlib.md5()
        with open(path, "rb") as f:
            cur.update(f.read())
        cur_hex = cur.hexdigest()
        expected = checksums_before.get(split)
        if expected and cur_hex != expected:
            errors.append(f"[{split}] file v1 {path} DA BI THAY DOI sau materialize "
                           f"(checksum {expected} -> {cur_hex})")
    return errors


def check_duplicate_change_ids(valid_changes: list, valid_merges: list, valid_removals: list) -> list:
    ids = [c.change_id for c in valid_changes] + [m.change_id for m in valid_merges] \
        + [r.change_id for r in valid_removals]
    seen, dups = set(), set()
    for i in ids:
        if i in seen:
            dups.add(i)
        seen.add(i)
    return [f"duplicate change_id={d}" for d in sorted(dups)]


# ── CSV writers (changelog) ────────────────────────────────────────────
def _write_csv(path: str, rows: list, fieldnames: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_skipped_csv(path: str, skipped: list) -> None:
    _write_csv(path, [
        {"source_workbook_row": s.source_workbook_row, "split": s.split,
         "sample_id": s.sample_id, "surface": s.surface,
         "reviewer_decision": s.reviewer_decision, "reason": s.reason}
        for s in skipped
    ], ["source_workbook_row", "split", "sample_id", "surface", "reviewer_decision", "reason"])


UNRESOLVED_DECISION_MARKERS = (
    "decision_not_auto_applicable:", "action_not_yet_implemented_for_auto_apply:",
    "unrecognized_decision_value:", "unrecognized_transaction_action:",
    "unrecognized_selected_action:", "TRANSACTION_ROLLED_BACK",
    "missing_structured_offset_or_split_spec_for:", "missing_structured_final_label_or_offset_fields",
)


def split_unresolved(skipped: list) -> list:
    """Subset cua skipped con can hanh dong trong tuong lai (khac voi
    no-op thuan tuy nhu KEEP_GOLD/rong)."""
    return [s for s in skipped if any(s.reason.startswith(m) or s.reason == m
                                       for m in UNRESOLVED_DECISION_MARKERS)]


# ── Top-level orchestration: ghi toàn bộ data/dataset_v2/* vào temp_root ──
def materialize_dataset_v2(
    temp_root: str, splits_v1: dict, source_map: dict,
    valid_changes: list, valid_merges: list, valid_removals: list,
    all_skipped: list, all_quarantined: list,
    materialization_config: dict, repo_dir: str,
    audit_manifest_path: Optional[str], current_checksums: dict,
    applied_review_workbooks: list, guideline_source_yaml: Optional[str],
    expected_quarantine_unique: Optional[int] = None,
    expected_total_sentences: Optional[int] = None,
) -> tuple:
    """
    Ghi toàn bộ cây thư mục output (xem hợp đồng phần A) vào temp_root.
    KHÔNG rename thành data/dataset_v2/ -- caller (apply_adjudication.py)
    tự quyết định rename sau khi hàm này trả success=True.

    Trả (success: bool, errors: list[str], report: dict).
    Nếu success=False, temp_root vẫn có thể chứa file (để debug) nhưng
    caller PHẢI KHÔNG rename nó thành output cuối.
    """
    errors = []

    dup_errs = check_duplicate_change_ids(valid_changes, valid_merges, valid_removals)
    errors.extend(dup_errs)

    new_splits, applied_entities, apply_errors, dedup_notes, dedup_records = apply_ops_to_splits(
        splits_v1, valid_changes, valid_merges, valid_removals,
    )
    errors.extend(apply_errors)
    if errors:
        return False, errors, {}

    verify_errs = verify_applied_effects(applied_entities, valid_changes, valid_merges, valid_removals)
    errors.extend(verify_errs)

    # BIO + overlap validation on v2_full
    bio_errors_rows = []
    for split, sents in new_splits.items():
        for idx, (tokens, labels) in enumerate(sents):
            for e in validate_bio_labels(tokens, labels):
                bio_errors_rows.append({"split": split, "sample_id": idx, "error": e})
            ents = bio_to_entities(tokens, labels)
            for e in check_no_flat_overlap(ents):
                bio_errors_rows.append({"split": split, "sample_id": idx, "error": f"overlap: {e}"})
    if bio_errors_rows:
        errors.extend(f"[{r['split']}:{r['sample_id']}] {r['error']}" for r in bio_errors_rows)

    if errors:
        return False, errors, {}

    quarantine_ids = quarantine_sample_ids_by_split(all_quarantined)
    clean_splits, index_map = build_clean_splits(new_splits, quarantine_ids)
    quarantine_records = build_quarantine_records(splits_v1, quarantine_ids, all_quarantined)

    dup_sample_errs = verify_no_duplicate_sample_ids(new_splits)
    partition_errs = verify_quarantine_partition(new_splits, clean_splits, quarantine_ids)
    add_entity_errs = verify_add_entity_never_applied(valid_changes, valid_merges, valid_removals)
    errors.extend(dup_sample_errs + partition_errs + add_entity_errs)

    # ── Contract assertions tuong minh theo yeu cau (khong suy doan) ────
    quarantine_unique_actual = sum(len(v) for v in quarantine_ids.values())
    if expected_quarantine_unique is not None and quarantine_unique_actual != expected_quarantine_unique:
        errors.append(
            f"CONTRACT VIOLATION: len(unique_quarantine_ids) = {quarantine_unique_actual}, "
            f"ky vong {expected_quarantine_unique}"
        )
    clean_ids_by_split = {split: set(range(len(sents))) for split, sents in clean_splits.items()}
    # index_map[split] anh xa old_idx (khong bi quarantine) -> new_idx trong
    # v2_clean; clean_ids o day dung KHONG GIAN old_idx (v2_full) de so sanh
    # truc tiep voi quarantine_ids (cung khong gian).
    clean_old_ids_by_split = {split: set(idx_map.keys()) for split, idx_map in index_map.items()}
    intersection = {
        split: clean_old_ids_by_split.get(split, set()) & quarantine_ids.get(split, set())
        for split in new_splits
    }
    nonempty_intersection = {k: v for k, v in intersection.items() if v}
    if nonempty_intersection:
        errors.append(f"CONTRACT VIOLATION: v2_clean_ids ∩ unique_quarantine_ids KHONG rong: {nonempty_intersection}")

    total_sentences_actual = sum(len(sents) for sents in new_splits.values())
    if expected_total_sentences is not None and total_sentences_actual != expected_total_sentences:
        errors.append(
            f"CONTRACT VIOLATION: dataset_v2_full co {total_sentences_actual} records, "
            f"ky vong {expected_total_sentences}"
        )

    if errors:
        return False, errors, {}

    # sample vua quarantine vua co op ap dung trong cung cau -- cho phep,
    # log ro (v2_full giu correction, v2_clean van loai het ca cau).
    touched_samples = {(op.split, s) for lst in (valid_changes, valid_merges, valid_removals)
                        for op in lst for s in [op.sample_id]}
    overlap_notes = sorted(
        f"{split}:{sid}" for split, sid in touched_samples
        if sid in quarantine_ids.get(split, set())
    )

    # ── Ghi dataset_v2_full & dataset_v2_clean ───────────────────────────
    full_full_stats = entity_stats(new_splits)
    clean_stats = entity_stats(clean_splits)
    n_changes_written = len(valid_changes) - sum(1 for r in dedup_records if r["dropped_action"] != "MERGE_ENTITY")
    n_merges_written = len(valid_merges) - sum(1 for r in dedup_records if r["dropped_action"] == "MERGE_ENTITY")
    quarantine_decision_rows = len(all_quarantined)
    quarantine_unique_records = sum(len(v) for v in quarantine_ids.values())
    applied_counts = {
        "n_changes_reviewlevel": len(valid_changes), "n_merges_reviewlevel": len(valid_merges),
        "n_changes_written": n_changes_written, "n_merges_written": n_merges_written,
        "n_removals": len(valid_removals),
        "gr11_changes": sum(1 for c in valid_changes if str(c.rule_id).startswith("GR-11")),
        "gr11_merges": sum(1 for m in valid_merges if str(m.rule_id).startswith("GR-11")),
    }

    for variant, sents_by_split, stats in (
        ("dataset_v2_full", new_splits, full_full_stats),
        ("dataset_v2_clean", clean_splits, clean_stats),
    ):
        vroot = os.path.join(temp_root, variant)
        for split, sents in sents_by_split.items():
            write_conll(os.path.join(vroot, f"{split}.txt"), sents)
            records = build_jsonl_records(sents, split, applied_entities, source_map, "v2")
            write_jsonl(os.path.join(vroot, f"{split}.jsonl"), records)
        with open(os.path.join(vroot, "dataset_card.md"), "w", encoding="utf-8") as f:
            f.write(render_dataset_card(variant, stats, applied_counts,
                                         quarantine_decision_rows, quarantine_unique_records))
        with open(os.path.join(vroot, "label_schema.json"), "w", encoding="utf-8") as f:
            json.dump(label_schema_dict(), f, ensure_ascii=False, indent=2)
        with open(os.path.join(vroot, "normalization_config.yaml"), "w", encoding="utf-8") as f:
            f.write(normalization_config_yaml())
        with open(os.path.join(vroot, "guideline_version.txt"), "w", encoding="utf-8") as f:
            f.write("v2.0\n")
        manifest = {
            "dataset_version": "v2", "variant": variant,
            "generated_at_utc": utc_now_iso(), "git_commit": git_commit_hash(repo_dir),
            "split_statistics": stats,
            "applied_counts": applied_counts,
            "quarantine_decision_rows_total": quarantine_decision_rows if variant == "dataset_v2_full" else None,
            "quarantined_unique_records_by_split": {k: len(v) for k, v in quarantine_ids.items()}
            if variant == "dataset_v2_full" else None,
            "quarantined_unique_records_total": quarantine_unique_records if variant == "dataset_v2_full" else None,
            "clean_index_map_old_to_new": index_map if variant == "dataset_v2_clean" else None,
        }
        with open(os.path.join(vroot, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        checksums = compute_dir_checksums(vroot)
        with open(os.path.join(vroot, "checksums.json"), "w", encoding="utf-8") as f:
            json.dump(checksums, f, ensure_ascii=False, indent=2)

    # ── Ghi quarantine_v2 ─────────────────────────────────────────────
    qroot = os.path.join(temp_root, "quarantine_v2")
    for split, recs in quarantine_records.items():
        write_jsonl(os.path.join(qroot, f"{split}_partial_annotation.jsonl"), recs)
    quarantine_manifest_rows = [
        {"split": split, "sample_id": sid, "n_candidates": len(
            [r for r in quarantine_records.get(split, []) if r["sample_id"] == sid])}
        for split, ids in quarantine_ids.items() for sid in sorted(ids)
    ]
    _write_csv(os.path.join(qroot, "quarantine_manifest.csv"), quarantine_manifest_rows,
               ["split", "sample_id", "n_candidates"])
    write_skipped_csv(os.path.join(qroot, "quarantine_decisions.csv"), all_quarantined)
    with open(os.path.join(qroot, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# quarantine_v2\n\n"
            "Sample có candidate GR-11 (date-formula) với "
            "`current_gold_configuration=unannotated` — KHÔNG có entity "
            "gold nào tại vị trí candidate, reviewer quyết định "
            "SKIP_UNANNOTATED_SAMPLE thay vì tự thêm nhãn.\n\n"
            "Giữ NGUYÊN raw text + label GỐC v1 (KHÔNG áp bất kỳ correction "
            "nào từ dataset_v2 cho các sample này trong thư mục này -- "
            "dataset_v2_full VẪN có thể chứa correction cho entity KHÁC "
            "trong cùng câu, xem manifest `overlap_with_applied_ops`).\n\n"
            "**KHÔNG dùng dữ liệu này cho supervised BIO NER training/"
            "evaluation.** suggested_action = COMPLETE_FULL_ANNOTATION_BEFORE_REUSE.\n\n"
            f"overlap_with_applied_ops (sample vừa quarantine vừa có op áp dụng ở chỗ khác "
            f"trong cùng câu, giữ ở v2_full, KHÔNG áp trong quarantine_v2 record): "
            f"{overlap_notes if overlap_notes else '(không có)'}\n"
        )
    with open(os.path.join(qroot, "checksums.json"), "w", encoding="utf-8") as f:
        json.dump(compute_dir_checksums(qroot), f, ensure_ascii=False, indent=2)

    # ── Ghi provenance ────────────────────────────────────────────────
    proot = os.path.join(temp_root, "provenance")
    os.makedirs(proot, exist_ok=True)
    with open(os.path.join(proot, "source_dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset_checksums_v1": current_checksums}, f, ensure_ascii=False, indent=2)
    audit_manifest_copy = {}
    if audit_manifest_path and os.path.exists(audit_manifest_path):
        with open(audit_manifest_path, encoding="utf-8") as f:
            audit_manifest_copy = json.load(f)
    with open(os.path.join(proot, "audit_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(audit_manifest_copy, f, ensure_ascii=False, indent=2)
    with open(os.path.join(proot, "applied_review_workbooks.json"), "w", encoding="utf-8") as f:
        json.dump(applied_review_workbooks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(proot, "materialization_config.json"), "w", encoding="utf-8") as f:
        json.dump(materialization_config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(proot, "git_commit.txt"), "w", encoding="utf-8") as f:
        f.write(git_commit_hash(repo_dir) + "\n")
    with open(os.path.join(proot, "run_timestamp_utc.txt"), "w", encoding="utf-8") as f:
        f.write(utc_now_iso() + "\n")
    with open(os.path.join(proot, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(environment_info())

    # ── Ghi changelog ─────────────────────────────────────────────────
    croot = os.path.join(temp_root, "changelog")
    _write_csv(os.path.join(croot, "applied_changes.csv"), [c.to_row() for c in valid_changes],
               list(valid_changes[0].to_row().keys()) if valid_changes else
               ["change_id", "sample_id", "split", "document_id", "rule_id", "reviewer_decision",
                "old_span", "old_label", "new_span", "new_label", "original_text",
                "normalized_text_if_any", "reviewer_notes", "source_workbook_row",
                "validation_status", "validation_message"])
    _write_csv(os.path.join(croot, "applied_merges.csv"), [m.to_row() for m in valid_merges],
               list(valid_merges[0].to_row().keys()) if valid_merges else
               ["change_id", "sample_id", "split", "document_id", "rule_id", "source_entity_ids",
                "source_spans", "resulting_span", "resulting_label", "reviewer_notes",
                "source_workbook_row", "validation_status", "validation_message"])
    _write_csv(os.path.join(croot, "applied_removals.csv"), [r.to_row() for r in valid_removals],
               list(valid_removals[0].to_row().keys()) if valid_removals else
               ["change_id", "sample_id", "split", "document_id", "rule_id", "action",
                "source_entity_id", "old_span", "old_label", "old_surface", "remove_reason",
                "reviewer_notes", "source_workbook_row", "linked_transaction_id",
                "validation_status", "validation_message"])
    write_skipped_csv(os.path.join(croot, "skipped_decisions.csv"), all_skipped)
    write_skipped_csv(os.path.join(croot, "quarantined_records.csv"), all_quarantined)
    write_skipped_csv(os.path.join(croot, "unresolved_decisions.csv"), split_unresolved(all_skipped))
    redundant_ops_fieldnames = [
        "dropped_operation_id", "kept_operation_id", "sample_id", "split",
        "dropped_action", "kept_action", "dropped_sources", "kept_sources",
        "final_target_span", "final_target_label", "dedup_reason",
        "source_workbook", "rule_id",
    ]
    _write_csv(os.path.join(croot, "redundant_ops_dropped.csv"),
               [{k: rec[k] for k in redundant_ops_fieldnames} for rec in dedup_records],
               redundant_ops_fieldnames)
    with open(os.path.join(croot, "summary.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Materialization changelog summary\n\n"
            f"Generated: {utc_now_iso()}\n\n"
            f"- Valid direct changes (review-level dry-run, truoc materializer dedup): {len(valid_changes)}\n"
            f"- Valid merges (review-level dry-run, truoc materializer dedup): {len(valid_merges)}\n"
            f"- Direct changes ACTUALLY WRITTEN (sau materializer redundancy dedup): {n_changes_written}\n"
            f"- Merges ACTUALLY WRITTEN (sau materializer redundancy dedup): {n_merges_written}\n"
            f"- Removals written: {len(valid_removals)}\n"
            f"- Redundant ops bi bo qua luc materialize (cung final target, old_ranges la tap con "
            f"cua 1 op khac -- 2 review khac nhau cung ket luan): {len(dedup_records)} "
            f"(xem redundant_ops_dropped.csv)\n"
            f"- Skipped decisions: {len(all_skipped)} (unresolved subset: {len(split_unresolved(all_skipped))})\n"
            f"- Quarantine decision rows (SKIP_UNANNOTATED_SAMPLE, KHONG phai so sample): {quarantine_decision_rows}\n"
            f"- Quarantined UNIQUE records (so sample_id thuc su bi loai khoi v2_clean): {quarantine_unique_records}\n"
            f"- Sample vừa quarantine vừa có op áp dụng ở chỗ khác cùng câu: {len(overlap_notes)}\n"
        )
    guideline_dir = os.path.join(temp_root, "guideline")
    os.makedirs(guideline_dir, exist_ok=True)
    with open(os.path.join(guideline_dir, "guideline_v2.0.md"), "w", encoding="utf-8") as f:
        f.write(guideline_v2_md(applied_counts))
    with open(os.path.join(guideline_dir, "guideline_v2.0.yaml"), "w", encoding="utf-8") as f:
        f.write(guideline_v2_yaml(guideline_source_yaml, applied_gr11=len(valid_changes) + len(valid_merges) > 0))

    # ── Ghi validation report (section H) ──────────────────────────────
    vroot = os.path.join(temp_root, "validation")
    os.makedirs(vroot, exist_ok=True)
    v1_stats = entity_stats(splits_v1)
    report = {
        "generated_at_utc": utc_now_iso(),
        "git_commit": git_commit_hash(repo_dir),
        "materialization_config": materialization_config,
        "record_counts": {
            "v1": {k: v["num_sentences"] for k, v in v1_stats.items()},
            "v2_full": {k: v["num_sentences"] for k, v in full_full_stats.items()},
            "v2_clean": {k: v["num_sentences"] for k, v in clean_stats.items()},
        },
        "entity_counts_before_after": {"v1": v1_stats, "v2_full": full_full_stats, "v2_clean": clean_stats},
        "applied_counts": applied_counts,
        "quarantine_decision_rows_by_split": _count_by_split([q.split for q in all_quarantined if q.split]),
        "quarantine_decision_rows_total": quarantine_decision_rows,
        "quarantined_unique_records_by_split": {k: len(v) for k, v in quarantine_ids.items()},
        "quarantined_unique_records_total": quarantine_unique_records,
        "skipped_count": len(all_skipped),
        "unresolved_count": len(split_unresolved(all_skipped)),
        "overlap_quarantine_and_applied_ops": overlap_notes,
        "redundant_ops_dropped_records": dedup_records,
        "redundant_ops_dropped_notes": dedup_notes,
        "bio_validation_errors": len(bio_errors_rows),
        "verify_applied_effects_errors": len(verify_errs),
        "duplicate_change_id_errors": len(dup_errs),
        "duplicate_sample_id_errors": len(dup_sample_errs),
        "quarantine_partition_errors": len(partition_errs),
        "add_entity_never_applied_errors": len(add_entity_errs),
        "example_sample_ids_by_change_type": {
            "change": [f"{c.split}:{c.sample_id}" for c in valid_changes[:5]],
            "merge": [f"{m.split}:{m.sample_id}" for m in valid_merges[:5]],
            "removal": [f"{r.split}:{r.sample_id}" for r in valid_removals[:5]],
        },
        "critical_samples": build_critical_samples_report(
            splits_v1, applied_entities,
            [("train", 437), ("train", 107), ("train", 948), ("train", 930),
             ("train", 18), ("train", 984), ("train", 7)],
        ),
    }
    with open(os.path.join(vroot, "post_apply_validation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_csv(os.path.join(vroot, "post_apply_errors.csv"),
               [{"error": e} for e in (verify_errs + dup_errs)], ["error"])

    split_stats_rows = []
    for split in splits_v1:
        split_stats_rows.append({
            "split": split, "num_sentences_v1": v1_stats[split]["num_sentences"],
            "num_sentences_v2_full": full_full_stats[split]["num_sentences"],
            "num_sentences_v2_clean": clean_stats[split]["num_sentences"],
        })
    _write_csv(os.path.join(vroot, "split_statistics_before_after.csv"), split_stats_rows,
               ["split", "num_sentences_v1", "num_sentences_v2_full", "num_sentences_v2_clean"])

    entity_stats_rows = []
    label_dist_rows = []
    for split in splits_v1:
        v1e = v1_stats[split]["num_entities"]
        v2e = full_full_stats[split]["num_entities"]
        vce = clean_stats[split]["num_entities"]
        entity_stats_rows.append({
            "split": split, "num_entities_v1": v1e, "num_entities_v2_full": v2e,
            "num_entities_v2_clean": vce,
        })
        labels = sorted(set(v1_stats[split]["label_distribution"]) | set(full_full_stats[split]["label_distribution"]))
        for lbl in labels:
            label_dist_rows.append({
                "split": split, "label": lbl,
                "count_v1": v1_stats[split]["label_distribution"].get(lbl, 0),
                "count_v2_full": full_full_stats[split]["label_distribution"].get(lbl, 0),
                "count_v2_clean": clean_stats[split]["label_distribution"].get(lbl, 0),
            })
    _write_csv(os.path.join(vroot, "entity_statistics_before_after.csv"), entity_stats_rows,
               ["split", "num_entities_v1", "num_entities_v2_full", "num_entities_v2_clean"])
    _write_csv(os.path.join(vroot, "label_distribution_before_after.csv"), label_dist_rows,
               ["split", "label", "count_v1", "count_v2_full", "count_v2_clean"])

    overlap_rows = []
    bio_rows = []
    normalized_offset_rows = []
    for split, sents in new_splits.items():
        for idx, (tokens, labels) in enumerate(sents):
            ents = bio_to_entities(tokens, labels)
            for e in check_no_flat_overlap(ents):
                overlap_rows.append({"split": split, "sample_id": idx, "detail": e})
            for e in validate_bio_labels(tokens, labels):
                bio_rows.append({"split": split, "sample_id": idx, "detail": e})
            text = "".join(tokens)
            for e in ents:
                ok = 0 <= e["start"] <= e["end"] < len(text)
                if not ok:
                    normalized_offset_rows.append({"split": split, "sample_id": idx,
                                                    "detail": f"offset out of range {e}"})
    _write_csv(os.path.join(vroot, "overlap_validation.csv"), overlap_rows, ["split", "sample_id", "detail"])
    _write_csv(os.path.join(vroot, "BIO_validation.csv"), bio_rows, ["split", "sample_id", "detail"])
    _write_csv(os.path.join(vroot, "normalized_offset_validation.csv"), normalized_offset_rows,
               ["split", "sample_id", "detail"])

    if verify_errs or dup_errs or bio_errors_rows:
        return False, verify_errs + dup_errs, report

    return True, [], report
