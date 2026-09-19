"""
audit_utils.py — annotation consistency audit (KHÔNG sửa nhãn gốc).

Mục tiêu: phát hiện và định lượng inconsistency trong gold label hiện có
(ORG<->TITLE, PER<->TITLE, LOC<->ORG, DTM, boundary địa danh hành chính)
để tạo review workbook cho người adjudicate — không tự động sửa, không
dùng để quyết định label.

Toàn bộ hàm ở đây thuần Python (list/dict), không phụ thuộc torch, để dễ
unit test và tái sử dụng độc lập với training pipeline.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

ENTITY_TYPES = ["PER", "LOC", "ORG", "TITLE", "DTM"]
LOC_ADMIN_SUFFIXES = ("州", "府", "營", "路", "鎮", "縣", "坊", "道", "社", "里")

# Các cặp nhãn dễ nhầm cần audit riêng (mục 5 yêu cầu)
CONFUSABLE_PAIRS = [
    ("ORG", "TITLE"),
    ("PER", "TITLE"),
    ("LOC", "ORG"),
    ("DTM", "PER"),
    ("DTM", "LOC"),
    ("DTM", "ORG"),
    ("DTM", "TITLE"),
]


# ── 1-2. BIO -> entity spans (parser thống nhất) ────────────────────────────
def bio_to_entities(tokens: list, labels: list) -> list:
    """
    Chuyển 1 câu (tokens, BIO labels) -> list[{start, end, type, violation}].
    start/end: chỉ số ký tự (inclusive) trong "".join(tokens).
    violation=True nếu entity bắt đầu bằng I-X mồ côi (không có B-X liền
    trước) -- đánh dấu, KHÔNG sửa; đây chính là 1 loại "candidate
    inconsistency" nên được liệt vào review, không phải lỗi code.
    Hỗ trợ cả BIOES (nếu labels có E-/S-) bằng cách coi E-X như I-X (đóng
    entity) và S-X như 1 entity 1 token -- dùng chung 1 hàm cho cả 2 scheme
    thay vì viết trùng logic (mục 2: "hỗ trợ BIO và BIOES nếu codebase hiện
    có cả hai").
    """
    n = len(tokens)
    entities = []
    i = 0
    while i < n:
        tag = labels[i]
        if tag == "O":
            i += 1
            continue
        prefix, etype = tag.split("-", 1)
        if prefix == "S":
            entities.append({"start": i, "end": i, "type": etype, "violation": False})
            i += 1
            continue
        violation = prefix == "I"  # I-X hoặc E-X mồ côi làm token mở đầu
        j = i + 1
        while j < n and labels[j] in (f"I-{etype}", f"E-{etype}"):
            is_end = labels[j] == f"E-{etype}"
            j += 1
            if is_end:
                break
        end = j - 1
        entities.append({"start": i, "end": end, "type": etype, "violation": violation})
        i = j
    return entities


def _file_checksum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_source_map(path: Optional[str]) -> dict:
    """
    record_source_map.json (repo ancient-chinese-ner, dataset DVSKTT v2.1)
    dạng list[{"split","idx","section",...}]. Trả {(split, idx): section}.
    Đã xác minh alignment (split,idx) <-> thứ tự câu trong dvsk_{split}.txt
    bằng cách so khớp n_chars — xem README audit mục "Assumptions". Trả
    dict rỗng nếu path=None hoặc file không tồn tại (document_id sẽ là None).
    """
    if not path or not os.path.exists(path):
        return {}
    import json
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {(r["split"], r["idx"]): r.get("section") for r in records}


def extract_all_spans(
    splits: dict, source_map: Optional[dict] = None, context_window: int = 20,
) -> list:
    """
    splits: {"train": [(tokens,labels), ...], "dev": [...], "test": [...]}
    Trả list[dict] — mỗi dict 1 entity, đủ field theo mục 2 của yêu cầu:
        entity_id, split, sample_id, document_id, sentence_id, text,
        start, end, surface, label, violation, context
    sample_id = sentence_id = thứ tự câu trong split (0-based) -- dữ liệu
    CoNLL char-level hiện tại không phân biệt 2 khái niệm này (1 record =
    1 câu), giữ cả 2 field để khớp schema yêu cầu.
    """
    source_map = source_map or {}
    records = []
    entity_id = 0
    for split, sentences in splits.items():
        for idx, (tokens, labels) in enumerate(sentences):
            text = "".join(tokens)
            document_id = source_map.get((split, idx))
            for ent in bio_to_entities(tokens, labels):
                s, e = ent["start"], ent["end"]
                ctx_start = max(0, s - context_window)
                ctx_end = min(len(text), e + 1 + context_window)
                surface = text[s:e + 1]
                marked_context = f"{text[ctx_start:s]}【{surface}】{text[e + 1:ctx_end]}"
                records.append({
                    "entity_id": entity_id,
                    "split": split,
                    "sample_id": idx,
                    "document_id": document_id,
                    "sentence_id": idx,
                    "text": text,
                    "start": s,
                    "end": e,
                    "surface": surface,
                    "label": ent["type"],
                    "violation": ent["violation"],
                    "context": text[ctx_start:ctx_end],
                    "marked_context": marked_context,
                })
                entity_id += 1
    return records


def normalize_surface(s: str) -> str:
    """Chuẩn hóa tối thiểu: chỉ strip khoảng trắng. Chữ Hán không có
    khái niệm case-folding; KHÔNG bỏ dấu câu/số vì có thể là 1 phần ý
    nghĩa thật của span (vd niên hiệu có số). Giới hạn này ghi trong
    README audit."""
    return s.strip()


# ── 3. Surface-form ambiguity report ────────────────────────────────────────
def build_ambiguity_report(entities: list) -> list:
    """Group theo normalized surface, chỉ giữ surface có >=2 nhãn khác
    nhau. Sort theo (số nhãn khác nhau giảm dần, tổng occurrence giảm dần)."""
    groups = defaultdict(list)
    for e in entities:
        groups[normalize_surface(e["surface"])].append(e)

    report = []
    for surface, occs in groups.items():
        labels = {o["label"] for o in occs}
        if len(labels) < 2:
            continue
        label_counts = defaultdict(int)
        for o in occs:
            label_counts[o["label"]] += 1
        report.append({
            "surface": surface,
            "total_occurrences": len(occs),
            "unique_labels": sorted(labels),
            "num_unique_labels": len(labels),
            "label_counts": dict(label_counts),
            "splits_present": sorted({o["split"] for o in occs}),
            "entity_lengths": sorted({o["end"] - o["start"] + 1 for o in occs}),
            "contexts": [o["context"] for o in occs],
            "sample_ids": [o["sample_id"] for o in occs],
            "document_ids": sorted({o["document_id"] for o in occs if o["document_id"]}),
            "entity_ids": [o["entity_id"] for o in occs],
        })

    report.sort(key=lambda r: (-r["num_unique_labels"], -r["total_occurrences"]))
    return report


# ── 4. Boundary consistency (hậu tố hành chính LOC) ─────────────────────────
def build_admin_suffix_boundary_report(entities: list) -> list:
    """
    Với entity LOC: stem = surface bỏ hậu tố hành chính nếu có, ngược lại
    giữ nguyên. Group theo stem; chỉ giữ stem xuất hiện CẢ 2 dạng (có và
    không có hậu tố) -- đây là candidate inconsistency, KHÔNG khẳng định
    gold sai (có thể cả 2 đều đúng theo ngữ cảnh, ví dụ 化州 vs 化 dùng như
    tỉnh lược hợp lệ trong văn bản gốc).
    """
    loc_entities = [e for e in entities if e["label"] == "LOC"]
    by_stem = defaultdict(lambda: {"with_suffix": [], "without_suffix": []})

    for e in loc_entities:
        surface = e["surface"]
        if surface and surface.endswith(LOC_ADMIN_SUFFIXES):
            stem = surface[:-1]
            by_stem[stem]["with_suffix"].append(e)
        else:
            by_stem[surface]["without_suffix"].append(e)

    report = []
    for stem, groups in by_stem.items():
        if not (groups["with_suffix"] and groups["without_suffix"]):
            continue
        occurrences = []
        for kind in ("with_suffix", "without_suffix"):
            for e in groups[kind]:
                occurrences.append({
                    "entity_id": e["entity_id"],
                    "split": e["split"],
                    "sample_id": e["sample_id"],
                    "gold_span": e["surface"],
                    "suffix_included": kind == "with_suffix",
                    "context": e["context"],
                    "label": e["label"],
                })
        report.append({
            "stem": stem,
            "num_with_suffix": len(groups["with_suffix"]),
            "num_without_suffix": len(groups["without_suffix"]),
            "occurrences": occurrences,
        })

    report.sort(key=lambda r: -(r["num_with_suffix"] + r["num_without_suffix"]))
    return report


# ── 4b. Merge/split boundary inconsistency (tổng quát hóa find_office_inconsistency.py) ──
_PUNCT = "。，、；：？！【】《》〈〉〔〕〕（）()."


def build_merge_split_report(entities: list, type_a: str, type_b: str) -> dict:
    """
    Tổng quát hóa script cũ scripts cục bộ find_office_inconsistency.py:
    phát hiện surface loại `type_a` vừa có bằng chứng bị TÁCH (đứng cạnh
    1 entity `type_b` liền kề, hoặc theo sau bởi 1 cụm ngắn ≤6 ký tự không
    gán nhãn) VỪA có bằng chứng bị GỘP (xuất hiện như tiền tố của 1 entity
    `type_b` dài hơn, trong CÙNG corpus). Trả candidates đã lọc theo cả 2
    bằng chứng -- đây là dấu hiệu guideline không nhất quán giữa các câu
    (không phải giữa 1 câu), KHÔNG khẳng định câu nào sai.
    """
    by_sentence = defaultdict(list)
    for e in entities:
        by_sentence[(e["split"], e["sample_id"])].append(e)

    type_b_surfaces = {normalize_surface(e["surface"]) for e in entities if e["label"] == type_b}

    split_adjacent = defaultdict(int)
    split_untagged = defaultdict(int)
    merged = defaultdict(set)

    for (split, sid), ents in by_sentence.items():
        ents_sorted = sorted(ents, key=lambda e: e["start"])
        text = ents_sorted[0]["text"] if ents_sorted else ""
        for i, e in enumerate(ents_sorted):
            if e["label"] != type_a:
                continue
            surface = normalize_surface(e["surface"])
            nxt = ents_sorted[i + 1] if i + 1 < len(ents_sorted) else None
            if nxt is not None and nxt["start"] == e["end"] + 1:
                if nxt["label"] == type_b:
                    split_adjacent[surface] += 1
                continue
            trailing_end = nxt["start"] if nxt is not None else len(text)
            trailing = text[e["end"] + 1: trailing_end]
            cut = next((k for k, ch in enumerate(trailing) if ch in _PUNCT), len(trailing))
            trailing_word = trailing[:cut]
            if 0 < len(trailing_word) <= 6:
                split_untagged[surface] += 1

    for e in entities:
        if e["label"] != type_a:
            continue
        stem = normalize_surface(e["surface"])
        for b_surf in type_b_surfaces:
            if b_surf.startswith(stem) and len(b_surf) > len(stem):
                merged[stem].add(b_surf)

    candidates = (set(split_adjacent) | set(split_untagged)) & set(merged)
    ranked = sorted(
        candidates,
        key=lambda s: -(split_adjacent[s] + split_untagged[s] + len(merged[s])),
    )
    return {
        "type_a": type_a,
        "type_b": type_b,
        "candidates": [
            {
                "surface": s,
                "split_adjacent_count": split_adjacent[s],
                "split_untagged_count": split_untagged[s],
                "merged_forms": sorted(merged[s])[:10],
            }
            for s in ranked
        ],
    }


# ── 5. Confusable-pair context report ───────────────────────────────────────
def build_confusable_pair_report(entities: list, pairs: list = None) -> list:
    """Với mỗi cặp (A,B), lấy các surface xuất hiện là A ở 1 chỗ và B ở
    chỗ khác -- tái dùng ambiguity report, lọc theo đúng 2 nhãn của cặp."""
    pairs = pairs or CONFUSABLE_PAIRS
    ambiguity = build_ambiguity_report(entities)
    amb_by_surface = {r["surface"]: r for r in ambiguity}

    report = []
    for type_a, type_b in pairs:
        matches = [
            r for r in ambiguity
            if type_a in r["unique_labels"] and type_b in r["unique_labels"]
        ]
        report.append({"pair": (type_a, type_b), "surfaces": matches})
    return report


# ── 8. Priority scoring (heuristic, không phải learned score) ──────────────
def compute_priority_score(
    num_unique_labels: int = 0,
    occurrence_count: int = 0,
    has_model_disagreement: bool = False,
    disagreement_confidence: float = 0.0,
    label: Optional[str] = None,
    is_boundary_admin_issue: bool = False,
    is_title_per_adjacency: bool = False,
    entity_length: int = 1,
    num_splits_present: int = 1,
) -> float:
    """Điểm ưu tiên HEURISTIC (không phải model học) theo đúng thứ tự mục
    8 yêu cầu -- dùng để SẮP XẾP review workbook, không phải quyết định
    đúng/sai. Trọng số chọn thủ công theo thứ tự ưu tiên liệt kê trong
    yêu cầu, có thể tinh chỉnh sau khi có phản hồi reviewer thật."""
    score = 0.0
    if num_unique_labels >= 3:
        score += 3.0
    elif num_unique_labels == 2:
        score += 2.0
    score += min(occurrence_count / 5.0, 2.0)
    if has_model_disagreement:
        score += 2.0 * disagreement_confidence
    if label in ("ORG", "DTM"):
        score += 2.0
    if is_boundary_admin_issue:
        score += 2.0
    if is_title_per_adjacency:
        score += 1.0
    if entity_length >= 2:
        score += 1.0
    if num_splits_present >= 2:
        score += 1.0
    return round(score, 3)


# ── 7. Review workbook (mỗi dòng = 1 decision unit / occurrence cụ thể) ─────
REVIEW_COLUMNS = [
    "review_id", "priority_score", "issue_type", "split", "sample_id",
    "document_id", "text", "marked_context", "gold_surface", "gold_start",
    "gold_end", "gold_label", "alternative_labels", "observed_labels_for_surface",
    "occurrence_count", "predicted_surface", "predicted_label",
    "predicted_confidence", "suggested_action", "reviewer_decision",
    "reviewer_notes", "final_label", "final_start", "final_end", "guideline_rule_id",
]


def build_review_workbook(
    entities: list,
    ambiguity_report: list,
    admin_suffix_report: list,
    merge_split_reports: dict,
    top_n: int = 300,
    model_disagreements: Optional[dict] = None,
) -> list:
    """
    Gộp candidate từ 3 nguồn (ambiguity / boundary admin-suffix /
    merge-split) thành 1 bảng review, mỗi dòng ứng với 1 occurrence cụ thể
    (dedup theo split+sample_id+start+end nếu cùng occurrence bị nhiều
    report gắn cờ — issue_type nối bằng ";"). reviewer_decision/
    reviewer_notes/final_* để TRỐNG — KHÔNG ghi đè dữ liệu gốc.
    model_disagreements: optional {(split, sample_id): {...}} nếu có chạy
    model disagreement (mục 6) — bỏ trống nếu không có checkpoint.
    """
    model_disagreements = model_disagreements or {}
    entity_by_id = {e["entity_id"]: e for e in entities}
    rows_by_key = {}

    def get_or_create(e):
        key = (e["split"], e["sample_id"], e["start"], e["end"])
        if key not in rows_by_key:
            rows_by_key[key] = {
                "split": e["split"], "sample_id": e["sample_id"],
                "document_id": e.get("document_id"), "text": e["text"],
                "marked_context": e["marked_context"], "gold_surface": e["surface"],
                "gold_start": e["start"], "gold_end": e["end"], "gold_label": e["label"],
                "issue_types": set(), "alternative_labels": set(),
                "observed_labels_for_surface": set(), "occurrence_count": 0,
                "notes": [], "priority_score": 0.0,
            }
        return rows_by_key[key]

    for r in ambiguity_report:
        for eid in r["entity_ids"]:
            e = entity_by_id[eid]
            row = get_or_create(e)
            row["issue_types"].add("label_ambiguity")
            row["alternative_labels"] |= (set(r["unique_labels"]) - {e["label"]})
            row["observed_labels_for_surface"] |= set(r["unique_labels"])
            row["occurrence_count"] = max(row["occurrence_count"], r["total_occurrences"])
            row["priority_score"] = max(row["priority_score"], compute_priority_score(
                num_unique_labels=r["num_unique_labels"], occurrence_count=r["total_occurrences"],
                label=e["label"], entity_length=e["end"] - e["start"] + 1,
                num_splits_present=len(r["splits_present"]),
            ))

    for r in admin_suffix_report:
        for occ in r["occurrences"]:
            e = entity_by_id[occ["entity_id"]]
            row = get_or_create(e)
            row["issue_types"].add("boundary_admin_suffix")
            row["notes"].append(
                f"stem='{r['stem']}' with_suffix={r['num_with_suffix']} "
                f"without_suffix={r['num_without_suffix']}"
            )
            row["priority_score"] = max(row["priority_score"], compute_priority_score(
                label=e["label"], is_boundary_admin_issue=True,
                entity_length=e["end"] - e["start"] + 1,
            ))

    for msr in merge_split_reports.values():
        type_a, type_b = msr["type_a"], msr["type_b"]
        cand_by_surface = {c["surface"]: c for c in msr["candidates"]}
        if not cand_by_surface:
            continue
        for e in entities:
            if e["label"] != type_a:
                continue
            surf = normalize_surface(e["surface"])
            c = cand_by_surface.get(surf)
            if c is None:
                continue
            row = get_or_create(e)
            row["issue_types"].add("merge_split_inconsistency")
            row["notes"].append(
                f"{type_a}->{type_b}: split_adjacent={c['split_adjacent_count']} "
                f"split_untagged={c['split_untagged_count']} merged_forms={c['merged_forms']}"
            )
            row["priority_score"] = max(row["priority_score"], compute_priority_score(
                label=e["label"], is_title_per_adjacency=("TITLE" in (type_a, type_b)),
                entity_length=e["end"] - e["start"] + 1,
            ))

    for key, dis in model_disagreements.items():
        for e in entities:
            if (e["split"], e["sample_id"]) != key:
                continue
            row = get_or_create(e)
            row["issue_types"].add("model_disagreement")
            row["priority_score"] = max(row["priority_score"], compute_priority_score(
                has_model_disagreement=True,
                disagreement_confidence=dis.get("confidence", 0.5),
                label=e["label"], entity_length=e["end"] - e["start"] + 1,
            ))

    ranked = sorted(rows_by_key.values(), key=lambda r: -r["priority_score"])[:top_n]
    rows = []
    for i, row in enumerate(ranked):
        rows.append({
            "review_id": i,
            "priority_score": row["priority_score"],
            "issue_type": ";".join(sorted(row["issue_types"])),
            "split": row["split"],
            "sample_id": row["sample_id"],
            "document_id": row["document_id"],
            "text": row["text"],
            "marked_context": row["marked_context"],
            "gold_surface": row["gold_surface"],
            "gold_start": row["gold_start"],
            "gold_end": row["gold_end"],
            "gold_label": row["gold_label"],
            "alternative_labels": ";".join(sorted(row["alternative_labels"])),
            "observed_labels_for_surface": ";".join(sorted(row["observed_labels_for_surface"])),
            "occurrence_count": row["occurrence_count"],
            "predicted_surface": "",
            "predicted_label": "",
            "predicted_confidence": "",
            "suggested_action": " | ".join(row["notes"]) or "requires adjudication",
            "reviewer_decision": "",
            "reviewer_notes": "",
            "final_label": "",
            "final_start": "",
            "final_end": "",
            "guideline_rule_id": "",
        })
    return rows


# ── Reviewer decision enum (mục 3, dùng chung mọi workbook deep-dive) ───────
REVIEWER_DECISION_OPTIONS = [
    "KEEP_GOLD",
    "CORRECT_LABEL",
    "CORRECT_BOUNDARY",
    "SPLIT_ENTITY",
    "MERGE_ENTITY",
    "GUIDELINE_AMBIGUITY",
    "NEEDS_DOMAIN_EXPERT",
    "UNSURE",
]

SEMANTIC_ROLE_OPTIONS = [
    "PERSON", "RULER", "DYNASTY", "CLAN", "POLITY", "PLACE",
    "OFFICE", "INSTITUTION", "TIME", "OTHER",
]


# ── Deep-dive review: singleton anomaly vs guideline ambiguity ─────────────
def find_singleton_anomalies(ambiguity_report: list, entities: list, min_occurrences: int = 10) -> list:
    """
    Nhóm A (mục 2): candidate lỗi gõ/lỗi nhãn cục bộ -- surface có
    >= min_occurrences occurrence, và có ĐÚNG 1 occurrence mang 1 nhãn
    khác hẳn phần còn lại (label count == 1). Đây KHÔNG khẳng định là lỗi
    -- chỉ là candidate có xác suất lỗi gõ cao hơn (đa số áp đảo ngược lại).
    Một surface có thể vừa xuất hiện ở đây (occurrence lẻ) vừa xuất hiện ở
    find_guideline_ambiguities() (phần đa số bị chia 2 nhóm lớn) -- 2 hiện
    tượng độc lập trên cùng 1 surface, ví dụ 光紹帝 (PER:15,TITLE:4,ORG:1):
    ORG=1 là singleton anomaly, PER-vs-TITLE=15-vs-4 là guideline ambiguity.
    """
    entity_by_id = {e["entity_id"]: e for e in entities}
    candidates = []
    for r in ambiguity_report:
        if r["total_occurrences"] < min_occurrences:
            continue
        for label, count in r["label_counts"].items():
            if count != 1:
                continue
            eid = next(eid for eid in r["entity_ids"] if entity_by_id[eid]["label"] == label)
            e = entity_by_id[eid]
            majority_label, majority_count = max(
                ((l, c) for l, c in r["label_counts"].items() if l != label),
                key=lambda kv: kv[1],
            )
            candidates.append({
                "surface": r["surface"],
                "minority_label": label,
                "minority_entity_id": eid,
                "minority_split": e["split"],
                "minority_sample_id": e["sample_id"],
                "minority_document_id": e.get("document_id"),
                "minority_context": e["marked_context"],
                "majority_label": majority_label,
                "majority_count": majority_count,
                "total_occurrences": r["total_occurrences"],
                "label_counts": r["label_counts"],
            })
    candidates.sort(key=lambda c: (-c["total_occurrences"], -c["majority_count"]))
    return candidates


def find_guideline_ambiguities(ambiguity_report: list, min_second_count: int = 2) -> list:
    """
    Nhóm B (mục 2): ambiguity mang tính hệ thống, KHÔNG coi minority là
    lỗi -- surface có >=2 nhãn mà nhãn PHỔ BIẾN THỨ NHÌ vẫn có
    >= min_second_count occurrence (tức không chỉ là 1 lần lẻ tẻ, mà là 1
    cách dùng thật sự cạnh tranh với nhãn phổ biến nhất). Đây là case cần
    RULE guideline rõ ràng, không phải sửa từng câu.
    """
    out = []
    for r in ambiguity_report:
        counts_sorted = sorted(r["label_counts"].values(), reverse=True)
        if len(counts_sorted) >= 2 and counts_sorted[1] >= min_second_count:
            out.append(r)
    out.sort(key=lambda r: -r["total_occurrences"])
    return out


def find_adjacent_split_variants(entities: list, compound_surface: str) -> list:
    """
    Kiểm tra xem `compound_surface` (vd "光紹帝") có bao giờ bị TÁCH thành
    2 entity liền kề ở 1 câu khác không (vd "光紹"+"帝" đứng sát nhau) --
    dấu hiệu segmentation không nhất quán giữa các câu.
    """
    by_sentence = defaultdict(list)
    for e in entities:
        by_sentence[(e["split"], e["sample_id"])].append(e)

    found = []
    for (split, sid), ents in by_sentence.items():
        ents_sorted = sorted(ents, key=lambda e: e["start"])
        for i in range(len(ents_sorted) - 1):
            a, b = ents_sorted[i], ents_sorted[i + 1]
            if b["start"] == a["end"] + 1 and (a["surface"] + b["surface"]) == compound_surface:
                found.append({
                    "split": split, "sample_id": sid,
                    "part1_surface": a["surface"], "part1_label": a["label"],
                    "part2_surface": b["surface"], "part2_label": b["label"],
                    "context": a["marked_context"],
                })
    return found


def get_neighboring_entities(entities: list, target: dict, window: int = 30) -> list:
    """Các entity KHÁC trong CÙNG câu với `target`, nằm trong bán kính
    `window` ký tự quanh span của target (không tính chính nó)."""
    neighbors = []
    for e in entities:
        if e["entity_id"] == target["entity_id"]:
            continue
        if e["split"] != target["split"] or e["sample_id"] != target["sample_id"]:
            continue
        if e["end"] < target["start"] - window or e["start"] > target["end"] + window:
            continue
        neighbors.append({"surface": e["surface"], "label": e["label"], "start": e["start"], "end": e["end"]})
    return sorted(neighbors, key=lambda n: n["start"])


def build_case_file_rows(entities: list, surface: str, context_window: int = 30) -> list:
    """
    Toàn bộ occurrence của 1 surface cụ thể (mục 1, 6) -- đủ field để
    reviewer xem full context + neighboring entities + kiểm tra span,
    KHÔNG suy diễn label/role.
    """
    norm = normalize_surface(surface)
    rows = []
    for e in entities:
        if normalize_surface(e["surface"]) != norm:
            continue
        ctx_start = max(0, e["start"] - context_window)
        ctx_end = min(len(e["text"]), e["end"] + 1 + context_window)
        marked = f"{e['text'][ctx_start:e['start']]}【{e['surface']}】{e['text'][e['end'] + 1:ctx_end]}"
        span_check_ok = e["text"][e["start"]: e["end"] + 1] == e["surface"]
        neighbors = get_neighboring_entities(entities, e, window=context_window)
        rows.append({
            "entity_id": e["entity_id"],
            "split": e["split"],
            "sample_id": e["sample_id"],
            "document_id": e.get("document_id"),
            "gold_surface": e["surface"],
            "gold_start": e["start"],
            "gold_end": e["end"],
            "gold_label": e["label"],
            "span_check_ok": span_check_ok,
            "full_sentence": e["text"],
            "context_pm30": marked,
            "neighboring_entities": "; ".join(f"{n['surface']}/{n['label']}" for n in neighbors),
            "semantic_role_candidate": "",  # reviewer điền, KHÔNG tự suy diễn
        })
    return rows
