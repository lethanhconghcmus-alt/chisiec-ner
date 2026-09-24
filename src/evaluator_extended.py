"""
evaluator_extended.py — bổ sung các metric benchmark R0/R1/R2 (SikuBERT +
Linear + CRF + BIO, KHÔNG boundary heads/BIOES/gazetteer) mà Evaluator
(evaluator.py, dùng chung cho M0 baseline) CHƯA có sẵn:

- lưu full test predictions per-sentence (schema tp/fp/fn theo (surface,
  type), TƯƠNG THÍCH với scripts/entity_length_unseen_f1_v2.py để tái sử
  dụng logic seen/unseen đã có, thay vì tự chế schema mới);
- partial-span/boundary error category counts (tái dùng
  src.evaluator_boundary._categorize_spans — logic ĐÃ có, không viết lại);
- ORG<->TITLE confusion cụ thể (cặp nhãn hay nhầm nhất theo
  audit_utils.CONFUSABLE_PAIRS);
- seen/unseen F1 theo (surface, type) có/không xuất hiện trong train;
- invalid BIO sequence count (find_bio_violations) + repair policy đã áp
  dụng khi trích entity (bio_to_entities: orphan I-X coi như mở entity mới,
  đánh dấu violation=True, KHÔNG raise, KHÔNG bỏ token);
- ví dụ GR-11 (DTM span dài, công thức ngày tháng hoàn chỉnh).

KHÔNG đổi Evaluator/Trainer cũ -- gọi THÊM sau khi train.py đã có
best checkpoint, dùng chung model/loader/id2label với pipeline hiện tại.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from src.audit_utils import CONFUSABLE_PAIRS, bio_to_entities
from src.bioes_utils import find_bio_violations
from src.evaluator_boundary import _categorize_spans
from src.utils import get_logger, save_json

logger = get_logger(__name__)

REPAIR_POLICY = (
    "Orphan I-X (khong co B-X/I-X lien truoc) duoc bio_to_entities() coi "
    "nhu MO MOT ENTITY MOI tai vi tri do (nhu the la B-X), danh dau "
    "violation=True -- KHONG raise, KHONG bo token, KHONG sua nhan du doan. "
    "Day la chinh sach 'repair-by-reinterpretation' dung thong nhat trong "
    "toan bo pipeline audit (xem src/audit_utils.py:bio_to_entities)."
)


@torch.no_grad()
def _run_inference(model, loader, id2label, device, constrained_crf=None):
    """-> list[(gold_tags, raw_pred_tags, constrained_pred_tags_or_None)]
    theo ĐÚNG thứ tự loader (loader PHẢI shuffle=False để khớp 1-1 với
    raw_data). raw_pred_tags LUÔN tính (decode gốc model.crf, unconstrained
    -- cho mục đích so sánh/diagnostic). constrained_pred_tags CHỈ tính khi
    truyền `constrained_crf` (xem src.constrained_decode) -- dùng chung 1
    lần forward backbone cho cả 2 decode (không tốn thêm 1 lần BERT forward)."""
    from src.constrained_decode import constrained_decode_batch
    model.eval()
    out = []
    for batch in tqdm(loader, desc="  extended eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"]
        gaz_ids = batch.get("gaz_ids")
        if gaz_ids is not None:
            gaz_ids = gaz_ids.to(device)
        raw_preds = model(input_ids, attention_mask, token_type_ids, gaz_ids=gaz_ids)
        if constrained_crf is not None:
            constrained_preds = constrained_decode_batch(model, constrained_crf, input_ids,
                                                           attention_mask, token_type_ids)
        else:
            constrained_preds = [None] * len(raw_preds)
        for raw_seq, cons_seq, label_seq, mask in zip(raw_preds, constrained_preds, labels, attention_mask):
            gold_tags, raw_tags, cons_tags = [], [], ([] if cons_seq is not None else None)
            for idx, (l, m) in enumerate(zip(label_seq.tolist(), mask.tolist())):
                if m == 0 or l == -100:
                    continue
                gold_tags.append(id2label[l])
                raw_tags.append(id2label[raw_seq[idx]])
                if cons_seq is not None:
                    cons_tags.append(id2label[cons_seq[idx]])
            out.append((gold_tags, raw_tags, cons_tags))
    return out


def _entity_spans(tokens: list, tags: list) -> list:
    """-> [(start, end, type, surface), ...]"""
    text = "".join(tokens)
    return [(e["start"], e["end"], e["type"], text[e["start"]:e["end"] + 1])
            for e in bio_to_entities(tokens, tags)]


def build_seen_surfaces(train_data: list) -> set:
    """{(surface, type), ...} từ toàn bộ entity gold trong train."""
    seen = set()
    for tokens, labels in train_data:
        for s, e, t, surface in _entity_spans(tokens, labels):
            seen.add((surface, t))
    return seen


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def extended_test_report(
    model, id2label: dict, device, output_dir: str, loader, raw_data: list,
    train_data: list, split: str = "test", gr11_min_len: int = 4, gr11_max_examples: int = 20,
    constrained_crf=None,
) -> dict:
    """
    raw_data: list[(tokens, labels)] GỐC (chưa tokenize/align subword), CÙNG
    THỨ TỰ với loader (loader phải shuffle=False, và max_len phải đủ lớn để
    không cắt câu -- nếu không, assert bên dưới sẽ chặn thay vì âm thầm lệch
    index).

    constrained_crf: None (mặc định, tương thích ngược) hoặc bản CRF đã hard-
    constrain (xem src.constrained_decode.build_constrained_crf). Khi truyền:
    - Entity extraction chính (tp/fp/fn, span category, ORG<->TITLE, seen/
      unseen, GR-11 examples) dùng CONSTRAINED tags (chính sách "production"
      -- xem artifacts/benchmark_v2/bio_decoding_policy.md).
    - RAW (unconstrained) tags vẫn được decode song song và đếm BIO violation
      riêng để so sánh/diagnostic (`bio_violations.raw_total_violations` vs
      `bio_violations.constrained_total_violations`).
    Khi None: hành vi CŨ (chỉ raw decode, dùng raw cho mọi thứ) -- giữ tương
    thích ngược cho caller không truyền tham số này.

    Ghi 2 file:
    - {split}_predictions_full.json: 1 record/câu, schema tp/fp/fn theo
      (surface, type) -- TƯƠNG THÍCH scripts/entity_length_unseen_f1_v2.py.
    - {split}_extended_report.json: aggregate (span category counts,
      ORG<->TITLE confusion, seen/unseen F1, BIO violations, GR-11 examples).

    Trả dict = nội dung {split}_extended_report.json.
    """
    infer = _run_inference(model, loader, id2label, device, constrained_crf=constrained_crf)
    if len(infer) != len(raw_data):
        raise ValueError(
            f"So cau infer ({len(infer)}) khac so cau raw_data ({len(raw_data)}) -- "
            f"loader co the bi shuffle=True hoac max_len cat mat cau, KHONG an toan "
            f"de zip 1-1. Kiem tra lai truoc khi tin ket qua nay."
        )

    seen_surfaces = build_seen_surfaces(train_data)

    predictions_full = []
    span_category_counts = {}
    org_title_confusion = {"gold_ORG_pred_TITLE": 0, "gold_TITLE_pred_ORG": 0}
    confusable_pair_counts = {f"gold_{a}_pred_{b}": 0 for a, b in CONFUSABLE_PAIRS} | \
                              {f"gold_{b}_pred_{a}": 0 for a, b in CONFUSABLE_PAIRS}
    raw_total_violations = 0
    raw_sentences_with_violations = 0
    constrained_total_violations = 0
    constrained_sentences_with_violations = 0
    gr11_examples = []

    all_gold_items, all_pred_items = [], []  # cho seen/unseen F1
    n_len_mismatch = 0

    for i, ((full_tokens, _labels_unused), (gold_tags, raw_tags, cons_tags)) in enumerate(zip(raw_data, infer)):
        pred_tags = cons_tags if cons_tags is not None else raw_tags
        # gold_tags/pred_tags co the NGAN HON full_tokens vi 2 ly do KHAC
        # NHAU (da xac minh thuc te qua smoke test voi SIKU-BERT/sikubert
        # tren du lieu that):
        #   (a) cau dai hon config.data.max_len - so special token -> bi cat
        #       O CUOI CAU (truncation that su, prefix alignment DUNG).
        #   (b) tokenizer AM THAM bo qua 1 vai ky tu PUA (Private Use Area,
        #       vd U+F0C47 -- bien the Han hiem, PHO BIEN trong corpus nay,
        #       xem docs PUA normalization) -- ky tu bi bo CO THE NAM O
        #       GIUA cau, KHONG chi o cuoi. Word_ids() cho tu do KHONG BAO
        #       GIO xuat hien, nen gold_tags/pred_tags ngan hon nhung KHONG
        #       phai la 1 prefix chinh xac cua full_tokens.
        # extended_test_report KHONG duoc truy nguoc word_ids() (NERDataset
        # khong expose ra ngoai), nen o day CHI align theo prefix (dung cho
        # case (a), XAP XI cho case (b) -- co the lam lech offset entity SAU
        # vi tri ky tu bi bo trong 1 so cau hiem). Metric CHINH (test_f1,
        # per-label P/R/F1 tu Evaluator.evaluate/seqeval) KHONG bi anh huong
        # boi han che nay vi seqeval so sanh THANG label sequence model thay,
        # khong can map nguoc ve ky tu. CHI cac enrichment o day (seen/
        # unseen surface, GR-11 example, ORG<->TITLE theo span) co the lech
        # nhe cho nhung cau hiem gap ky tu PUA giua cau.
        if len(gold_tags) != len(full_tokens):
            n_len_mismatch += 1
        tokens = full_tokens[:len(gold_tags)]
        text = "".join(tokens)
        gold_spans = _entity_spans(tokens, gold_tags)
        pred_spans = _entity_spans(tokens, pred_tags)

        gold_set = {(s, e, t) for s, e, t, _sf in gold_spans}
        pred_set = {(s, e, t) for s, e, t, _sf in pred_spans}
        tp_set = gold_set & pred_set
        tp = [(t, sf) for s, e, t, sf in gold_spans if (s, e, t) in tp_set]
        fn = [(t, sf) for s, e, t, sf in gold_spans if (s, e, t) not in tp_set]
        fp = [(t, sf) for s, e, t, sf in pred_spans if (s, e, t) not in tp_set]

        for surface_type in tp:
            item = {"type": surface_type[0], "surface": surface_type[1],
                    "seen": (surface_type[1], surface_type[0]) in seen_surfaces, "matched": True}
            all_gold_items.append(item)
            all_pred_items.append(item)
        for etype, surface in fn:
            all_gold_items.append({"type": etype, "surface": surface,
                                    "seen": (surface, etype) in seen_surfaces, "matched": False})
        for etype, surface in fp:
            all_pred_items.append({"type": etype, "surface": surface,
                                    "seen": (surface, etype) in seen_surfaces, "matched": False})

        cats = _categorize_spans([(s, e, t) for s, e, t, _ in gold_spans],
                                  [(s, e, t) for s, e, t, _ in pred_spans])
        for c in cats:
            span_category_counts[c] = span_category_counts.get(c, 0) + 1

        gold_by_span = {(s, e): t for s, e, t, _ in gold_spans}
        pred_by_span = {(s, e): t for s, e, t, _ in pred_spans}
        for (s, e), gt in gold_by_span.items():
            pt = pred_by_span.get((s, e))
            if pt is None or pt == gt:
                continue
            key = f"gold_{gt}_pred_{pt}"
            if key in confusable_pair_counts:
                confusable_pair_counts[key] += 1
            if gt == "ORG" and pt == "TITLE":
                org_title_confusion["gold_ORG_pred_TITLE"] += 1
            elif gt == "TITLE" and pt == "ORG":
                org_title_confusion["gold_TITLE_pred_ORG"] += 1

        raw_violations = find_bio_violations(raw_tags, sample_id=i)
        if raw_violations:
            raw_sentences_with_violations += 1
            raw_total_violations += len(raw_violations)
        if cons_tags is not None:
            constrained_violations = find_bio_violations(cons_tags, sample_id=i)
            if constrained_violations:
                constrained_sentences_with_violations += 1
                constrained_total_violations += len(constrained_violations)
        else:
            constrained_violations = None
        violations = constrained_violations if constrained_violations is not None else raw_violations

        for s, e, t, surface in gold_spans:
            if t == "DTM" and (e - s + 1) >= gr11_min_len and len(gr11_examples) < gr11_max_examples:
                pred_here = pred_by_span.get((s, e))
                gr11_examples.append({
                    "sample_id": i, "raw_text": text, "gold_span": [s, e],
                    "gold_surface": surface, "pred_label_at_span": pred_here,
                    "exact_match": pred_here == "DTM",
                })

        predictions_full.append({
            "sample_id": i, "raw_text": text, "tokens": tokens,
            "gold_tags": gold_tags, "pred_tags": pred_tags,
            "raw_pred_tags": raw_tags, "constrained_pred_tags": cons_tags,
            "tp": [[sf, t] for t, sf in tp], "fp": [[sf, t] for t, sf in fp],
            "fn": [[sf, t] for t, sf in fn],
            "bio_violations": violations,
            "raw_bio_violations": raw_violations,
            "constrained_bio_violations": constrained_violations,
        })

    by_seen = {}
    for flag, name in [(True, "seen"), (False, "unseen")]:
        n_gold = sum(1 for g in all_gold_items if g["seen"] == flag)
        n_pred = sum(1 for p in all_pred_items if p["seen"] == flag)
        tp_n = sum(1 for g in all_gold_items if g["seen"] == flag and g["matched"])
        p, r, f1 = _prf(tp_n, n_pred, n_gold)
        by_seen[name] = {"precision": p, "recall": r, "f1": f1,
                          "n_gold": n_gold, "n_pred": n_pred, "tp": tp_n}

    report = {
        "split": split,
        "n_sentences": len(raw_data),
        "n_sentences_with_subword_coverage_gap": n_len_mismatch,
        "subword_coverage_gap_note": (
            "So cau co len(model_output) != len(raw_tokens) -- 2 nguyen nhan: "
            "(a) that su bi cat boi data.max_len (gap o CUOI cau, alignment prefix "
            "chinh xac); (b) tokenizer bo qua ky tu PUA (Private Use Area, vd "
            "U+F0C47) khong co trong vocab -- gap co the o GIUA cau, alignment "
            "prefix o day chi la XAP XI. KHONG anh huong test_f1/per-label P/R/F1 "
            "chinh (seqeval so sanh thang label sequence, khong can toa do ky tu); "
            "CHI co the lam lech nhe cac truong enrichment duoi day (surface/span) "
            "cho nhung cau hiem gap giua cau."
        ),
        "span_category_counts": span_category_counts,
        "org_title_confusion": org_title_confusion,
        "confusable_pair_counts": {k: v for k, v in confusable_pair_counts.items() if v > 0},
        "seen_unseen_f1": by_seen,
        "decode_policy": "constrained" if constrained_crf is not None else "raw_unconstrained",
        "bio_violations": {
            "raw_total_violations": raw_total_violations,
            "raw_sentences_with_violations": raw_sentences_with_violations,
            "constrained_total_violations": constrained_total_violations if constrained_crf is not None else None,
            "constrained_sentences_with_violations": (
                constrained_sentences_with_violations if constrained_crf is not None else None
            ),
            "repair_policy": REPAIR_POLICY,
            "repair_policy_note": (
                "Repair-by-reinterpretation (bio_to_entities) CHI dung noi bo de trich "
                "entity cho cac truong enrichment o day khi decode_policy='raw_unconstrained' "
                "(constrained_crf=None). Khi decode_policy='constrained', KHONG can repair "
                "(constrained decode dam bao 0 violation by construction)."
            ),
        },
        "gr11_dtm_examples": gr11_examples,
    }

    save_json(predictions_full, f"{output_dir}/{split}_predictions_full.json")
    save_json(report, f"{output_dir}/{split}_extended_report.json")
    logger.info(f"Extended report saved -> {output_dir}/{split}_extended_report.json")
    if n_len_mismatch:
        logger.warning(
            f"[{split}] {n_len_mismatch}/{len(raw_data)} cau co subword coverage gap "
            f"(max_len truncation HOAC ky tu PUA bi tokenizer bo qua) -- xem "
            f"report['subword_coverage_gap_note']. KHONG anh huong test_f1 chinh."
        )
    return report
