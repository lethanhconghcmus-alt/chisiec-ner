"""
evaluator_boundary.py — strict entity-level evaluation cho BertCRFBoundaryNER
(scheme BIOES + boundary heads). KHÔNG đụng vào evaluator.py/trainer.py cũ
(dùng cho GuwenBertCRF/GuwenBertLinear — baseline M0, hoặc M1 qua
Evaluator(scheme="IOBES")). Đây là evaluator riêng cho M2 vì
BertCRFBoundaryNER có forward signature khác (ner_labels/start_labels/
end_labels thay vì labels, trả về BoundaryNEROutput thay vì tuple).
"""

import json
from collections import defaultdict

import torch
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOBES

from src.utils import get_logger, save_json
from src.bioes_utils import bioes_to_spans, convert_bio_to_bioes, repair_bioes_sequence

logger = get_logger(__name__)

# Mục F: LOC hành chính kết thúc bằng các hậu tố này
LOC_ADMIN_SUFFIXES = ("州", "府", "營", "路", "鎮", "縣", "坊")


def _binary_prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _tags_to_spans_any_scheme(tags: list, label_scheme: str) -> tuple:
    """Trả (repaired_tags_bioes_form, spans, num_repairs). Với scheme='bio'
    (M3), decode qua torchcrf đã hard-constrained (illegal_start cấm I-* mở
    đầu, illegal_transition cấm orphan-I/khác-loại — xem
    bioes_utils.build_transition_masks(scheme='bio')) nên KHÔNG cần repair
    thật, nhưng vẫn convert sang BIOES (mode='repair', phòng hờ) để dùng
    LẠI đúng 1 hàm bioes_to_spans() cho cả 2 scheme thay vì viết trùng logic
    parse span riêng cho BIO."""
    if label_scheme == "bio":
        bioes_tags, n_rep = convert_bio_to_bioes(tags, mode="repair")
        return bioes_tags, bioes_to_spans(bioes_tags), n_rep
    repaired, n_rep = repair_bioes_sequence(tags)
    return repaired, bioes_to_spans(repaired), n_rep


class BoundaryEvaluator:
    def __init__(self, model, id2label: dict, device: torch.device, output_dir: str,
                 label_scheme: str = "bioes"):
        """label_scheme: 'bioes' (mặc định, M2) hoặc 'bio' (M3) — PHẢI khớp
        đúng scheme model thật sự dùng, quyết định cách parse
        decoded_tags -> spans và cách gọi seqeval (strict IOBES vs BIO
        mặc định)."""
        if label_scheme not in ("bio", "bioes"):
            raise ValueError(f"Unknown label_scheme: {label_scheme!r}")
        self.model = model
        self.id2label = id2label
        self.device = device
        self.output_dir = output_dir
        self.label_scheme = label_scheme

    # ── forward 1 batch, KHÔNG truyền labels -> CRF decode + boundary logits ──
    @torch.no_grad()
    def _forward_decode(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        return self.model(input_ids, attention_mask, token_type_ids), attention_mask

    # ── CORE EVALUATE (dùng để chọn checkpoint mỗi epoch) ───────────────────
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """
        strict entity-level P/R/F1 (mode=strict, scheme=IOBES) qua CRF decode
        (V1: boundary heads KHÔNG post-process CRF output — mục F) + boundary
        head P/R/F1 (threshold sigmoid>0.5, chỉ trên valid_ner_mask — mục C).
        """
        self.model.eval()
        all_gold_tags, all_pred_tags = [], []
        start_tp = start_fp = start_fn = 0
        end_tp = end_fp = end_fn = 0
        total_repairs = 0

        for batch in tqdm(loader, desc="  eval(boundary)", leave=False, dynamic_ncols=True):
            out, attention_mask = self._forward_decode(batch)
            labels = batch["labels"]

            for pred_seq, label_seq in zip(out.decoded_tags, labels):
                gold_tags = [self.id2label[l.item()] for l in label_seq if l.item() != -100]
                pred_tags_full = [self.id2label[p] for p in pred_seq]
                # decode length = số vị trí valid trong CRF mask, mask[:,0]
                # luôn True (bao gồm CLS) — bỏ phần tử đầu để căn theo
                # gold_tags (chỉ gồm token thật, không CLS/SEP/pad).
                pred_tags = pred_tags_full[1: 1 + len(gold_tags)]
                if self.label_scheme == "bioes":
                    pred_tags, n_rep = repair_bioes_sequence(pred_tags)
                    total_repairs += n_rep
                # scheme "bio": không repair -- hard-constrained CRF (scheme
                # "bio") đã loại orphan-I ở decode, và seqeval mặc định (BIO)
                # tự xử lý an toàn không cần repair, giống hệt Evaluator cũ
                # (M0/M1).
                all_gold_tags.append(gold_tags)
                all_pred_tags.append(pred_tags)

            if "start_labels" in batch and "end_labels" in batch:
                valid_ner_mask = attention_mask.bool() & labels.to(self.device).ne(-100)
                start_pred = (out.start_logits > 0).long()
                end_pred = (out.end_logits > 0).long()
                start_gold = batch["start_labels"].to(self.device)
                end_gold = batch["end_labels"].to(self.device)

                sp, sg = start_pred[valid_ner_mask], start_gold[valid_ner_mask]
                start_tp += int(((sp == 1) & (sg == 1)).sum())
                start_fp += int(((sp == 1) & (sg == 0)).sum())
                start_fn += int(((sp == 0) & (sg == 1)).sum())

                ep, eg = end_pred[valid_ner_mask], end_gold[valid_ner_mask]
                end_tp += int(((ep == 1) & (eg == 1)).sum())
                end_fp += int(((ep == 1) & (eg == 0)).sum())
                end_fn += int(((ep == 0) & (eg == 1)).sum())

        if total_repairs:
            logger.warning(
                f"[BoundaryEvaluator] repaired {total_repairs} invalid BIOES "
                f"transition(s) từ CRF decode trước khi tính spans (torchcrf "
                f"không enforce hard transition constraint — xem "
                f"bioes_utils.repair_bioes_sequence)."
            )

        seqeval_kwargs = {"mode": "strict", "scheme": IOBES} if self.label_scheme == "bioes" else {}
        report = classification_report(
            all_gold_tags, all_pred_tags, output_dict=True, zero_division=0, **seqeval_kwargs,
        )
        f1 = f1_score(all_gold_tags, all_pred_tags, zero_division=0, **seqeval_kwargs)

        sp_, sr_, sf1_ = _binary_prf(start_tp, start_fp, start_fn)
        ep_, er_, ef1_ = _binary_prf(end_tp, end_fp, end_fn)

        return {
            "f1": f1,
            "report": report,
            "num_bioes_repairs": total_repairs,
            "start_precision": sp_, "start_recall": sr_, "start_f1": sf1_,
            "end_precision": ep_, "end_recall": er_, "end_f1": ef1_,
        }

    # ── FULL REPORT (log + save json) ───────────────────────────────────────
    def full_report(self, loader, split: str = "test") -> dict:
        metrics = self.evaluate(loader)
        logger.info(f"\n{'─'*55}")
        logger.info(f"[{split.upper()}] Strict entity F1 = {metrics['f1']:.4f}")
        for ent, sc in metrics["report"].items():
            if isinstance(sc, dict):
                logger.info(
                    f"  {ent:12s} P={sc['precision']:.3f}  R={sc['recall']:.3f}  "
                    f"F1={sc['f1-score']:.3f}  (support={sc['support']})"
                )
        logger.info(
            f"  start P={metrics['start_precision']:.3f} R={metrics['start_recall']:.3f} "
            f"F1={metrics['start_f1']:.3f} | end P={metrics['end_precision']:.3f} "
            f"R={metrics['end_recall']:.3f} F1={metrics['end_f1']:.3f}"
        )
        logger.info(f"{'─'*55}")
        save_json(metrics, f"{self.output_dir}/{split}_report.json")
        return metrics

    # ── mục F.8/F.9: error analysis + JSONL export ──────────────────────────
    @torch.no_grad()
    def error_analysis(self, loader, raw_data: list, split: str = "test") -> dict:
        """
        raw_data: list[(tokens, bioes_labels)] cùng thứ tự với loader
        (shuffle=False). Ghi <output_dir>/<split>_errors.jsonl theo đúng
        field list mục F.9, và 1 summary dict trả về (+ lưu json riêng).
        """
        self.model.eval()
        categories = defaultdict(int)
        records = []
        idx = 0

        for batch in tqdm(loader, desc="  error analysis(boundary)", leave=False):
            out, attention_mask = self._forward_decode(batch)
            labels = batch["labels"]
            bsz = labels.shape[0]

            start_gold_batch = batch.get("start_labels")
            end_gold_batch = batch.get("end_labels")

            for b in range(bsz):
                tokens, gold_bioes_full = raw_data[idx]
                idx += 1

                label_seq = labels[b]
                gold_tags = [self.id2label[l.item()] for l in label_seq if l.item() != -100]
                pred_tags_full = [self.id2label[p] for p in out.decoded_tags[b]]
                pred_tags = pred_tags_full[1: 1 + len(gold_tags)]
                pred_tags_repaired, pred_spans, _ = _tags_to_spans_any_scheme(pred_tags, self.label_scheme)
                gold_tags_repaired, gold_spans, _ = _tags_to_spans_any_scheme(gold_tags, self.label_scheme)

                sample_categories = _categorize_spans(gold_spans, pred_spans)
                for c in sample_categories:
                    categories[c] += 1

                start_pred = (out.start_logits[b] > 0).long().tolist() if start_gold_batch is not None else None
                end_pred = (out.end_logits[b] > 0).long().tolist() if end_gold_batch is not None else None

                records.append({
                    "sample_id": idx - 1,
                    "raw_text": "".join(tokens),
                    "tokens": tokens,
                    "gold_BIOES": gold_tags_repaired,
                    "pred_BIOES": pred_tags_repaired,
                    "gold_entities": [
                        {"text": "".join(tokens[s:e + 1]), "start": s, "end": e, "type": t}
                        for s, e, t in gold_spans
                    ],
                    "pred_entities": [
                        {"text": "".join(tokens[s:e + 1]), "start": s, "end": e, "type": t}
                        for s, e, t in pred_spans
                    ],
                    "start_gold": start_gold_batch[b].tolist() if start_gold_batch is not None else None,
                    "start_pred": start_pred,
                    "end_gold": end_gold_batch[b].tolist() if end_gold_batch is not None else None,
                    "end_pred": end_pred,
                    "error_categories": sorted(sample_categories) if sample_categories else [],
                })

        out_path = f"{self.output_dir}/{split}_errors.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Error report saved → {out_path} ({len(records)} samples)")

        # ── Báo cáo riêng: PER/LOC len>=2, LOC hậu tố hành chính, partial-span ──
        focus = _build_focus_report(records)
        summary = {
            "error_category_counts": dict(categories),
            "total_samples": len(records),
            "focus_report": focus,
        }
        save_json(summary, f"{self.output_dir}/{split}_error_summary.json")
        logger.info(f"Error summary saved → {self.output_dir}/{split}_error_summary.json")
        return summary


def _categorize_spans(gold_spans: list, pred_spans: list) -> set:
    """Mục F.8 — phân loại lỗi ở mức câu (set các category xuất hiện)."""
    cats = set()
    gold_set, pred_set = set(gold_spans), set(pred_spans)
    exact_match = gold_set & pred_set
    if exact_match:
        cats.add("exact_span_correct_type")

    gold_by_span = {(s, e): t for s, e, t in gold_spans}
    pred_by_span = {(s, e): t for s, e, t in pred_spans}

    for (s, e), t in gold_by_span.items():
        if (s, e) in pred_by_span:
            if pred_by_span[(s, e)] != t:
                cats.add("wrong_type_exact_boundary")
            continue
        # tìm pred overlap cùng loại
        matched = False
        for ps, pe, pt in pred_spans:
            if pt != t:
                continue
            if ps == s and pe < e:
                cats.add("predicted_is_prefix")
                matched = True
            elif pe == e and ps > s:
                cats.add("predicted_is_suffix")
                matched = True
            elif ps <= s and pe >= e and (ps, pe) != (s, e):
                cats.add("predicted_longer")
                matched = True
            elif not (pe < s or ps > e):  # overlap khác kiểu
                cats.add("correct_type_wrong_boundary")
                matched = True
        if not matched:
            cats.add("missed_entity")

    for (s, e), t in pred_by_span.items():
        if (s, e) not in gold_by_span:
            overlaps_gold = any(not (ge < s or gs > e) for gs, ge, gt in gold_spans)
            if not overlaps_gold:
                cats.add("spurious_entity")

    return cats


def _build_focus_report(records: list) -> dict:
    per_loc_len2 = []
    loc_admin_suffix = []
    partial_span_examples = []

    for rec in records:
        gold_by_span = {(e["start"], e["end"]): e["text"] for e in rec["gold_entities"]}
        pred_by_span = {(e["start"], e["end"]): e["text"] for e in rec["pred_entities"]}

        for e in rec["gold_entities"]:
            if e["type"] in ("PER", "LOC") and len(e["text"]) >= 2:
                found_exact = (e["start"], e["end"]) in pred_by_span
                per_loc_len2.append({
                    "sample_id": rec["sample_id"], "type": e["type"], "gold": e["text"],
                    "correct": found_exact,
                })
            if e["type"] == "LOC" and e["text"].endswith(LOC_ADMIN_SUFFIXES):
                found_exact = (e["start"], e["end"]) in pred_by_span
                loc_admin_suffix.append({
                    "sample_id": rec["sample_id"], "gold": e["text"], "correct": found_exact,
                })

        # partial-span: pred là prefix/suffix của 1 gold cùng loại, không exact
        pred_spans_by_type = {}
        for e in rec["pred_entities"]:
            pred_spans_by_type.setdefault(e["type"], []).append(e)
        for e in rec["gold_entities"]:
            if (e["start"], e["end"]) in pred_by_span:
                continue
            for pe in pred_spans_by_type.get(e["type"], []):
                if pe["start"] == e["start"] and pe["end"] < e["end"]:
                    partial_span_examples.append({
                        "sample_id": rec["sample_id"], "gold": e["text"], "pred": pe["text"],
                        "kind": "prefix",
                    })
                elif pe["end"] == e["end"] and pe["start"] > e["start"]:
                    partial_span_examples.append({
                        "sample_id": rec["sample_id"], "gold": e["text"], "pred": pe["text"],
                        "kind": "suffix",
                    })

    def _acc(items):
        return sum(1 for i in items if i["correct"]) / len(items) if items else None

    return {
        "per_loc_len_ge2": {"count": len(per_loc_len2), "accuracy": _acc(per_loc_len2),
                             "samples": per_loc_len2[:30]},
        "loc_admin_suffix": {"count": len(loc_admin_suffix), "accuracy": _acc(loc_admin_suffix),
                              "samples": loc_admin_suffix[:30]},
        "partial_span_examples": partial_span_examples[:30],
    }
