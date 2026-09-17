import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.bioes_utils import build_bioes_label_map
from src.data_utils import NERDataset
from src.evaluator_boundary import BoundaryEvaluator, _categorize_spans
from src.models import BertCRFBoundaryNER

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


# ── pure-function span categorization ───────────────────────────────────────
def test_categorize_spans_exact_match():
    gold = [(0, 1, "PER")]
    pred = [(0, 1, "PER")]
    assert _categorize_spans(gold, pred) == {"exact_span_correct_type"}


def test_categorize_spans_wrong_type_exact_boundary():
    gold = [(0, 1, "PER")]
    pred = [(0, 1, "LOC")]
    assert _categorize_spans(gold, pred) == {"wrong_type_exact_boundary"}


def test_categorize_spans_prefix():
    # gold 黎時憲 (0..2), pred 黎時 (0..1) — pred là prefix của gold
    gold = [(0, 2, "PER")]
    pred = [(0, 1, "PER")]
    assert _categorize_spans(gold, pred) == {"predicted_is_prefix"}


def test_categorize_spans_suffix():
    gold = [(0, 2, "LOC")]
    pred = [(1, 2, "LOC")]
    assert _categorize_spans(gold, pred) == {"predicted_is_suffix"}


def test_categorize_spans_predicted_longer():
    gold = [(1, 2, "LOC")]
    pred = [(0, 3, "LOC")]
    assert _categorize_spans(gold, pred) == {"predicted_longer"}


def test_categorize_spans_missed_entity():
    gold = [(0, 1, "PER")]
    pred = []
    assert _categorize_spans(gold, pred) == {"missed_entity"}


def test_categorize_spans_spurious_entity():
    gold = []
    pred = [(0, 1, "PER")]
    assert _categorize_spans(gold, pred) == {"spurious_entity"}


# ── integration: BoundaryEvaluator trên model đã train ngắn ────────────────
def _build_tiny_trained_model_and_loader():
    torch.manual_seed(0)
    label2id, id2label = build_bioes_label_map()
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)

    data = [
        (list("太師陳守度至化州"),
         ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "B-LOC", "E-LOC"]),
        (list("以黄義膠為督視"),
         ["O", "B-PER", "I-PER", "E-PER", "O", "B-TITLE", "E-TITLE"]),
    ] * 10

    ds = NERDataset(data, tokenizer, label2id, max_len=16, derive_boundary=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    model = BertCRFBoundaryNER(TINY_BACKBONE, num_labels=len(label2id),
                                o_label_id=label2id["O"], boundary_loss_type="bce")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(15):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                        ner_labels=batch["labels"], start_labels=batch["start_labels"],
                        end_labels=batch["end_labels"])
            out.loss.backward()
            optimizer.step()

    return model, id2label, eval_loader, data


def test_boundary_evaluator_evaluate_returns_valid_metrics():
    model, id2label, loader, _ = _build_tiny_trained_model_and_loader()
    evaluator = BoundaryEvaluator(model, id2label, torch.device("cpu"), output_dir=".")
    metrics = evaluator.evaluate(loader)

    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["start_f1"] <= 1.0
    assert 0.0 <= metrics["end_f1"] <= 1.0
    assert "report" in metrics


def test_boundary_evaluator_error_analysis_writes_jsonl(tmp_path):
    model, id2label, loader, data = _build_tiny_trained_model_and_loader()
    evaluator = BoundaryEvaluator(model, id2label, torch.device("cpu"), output_dir=str(tmp_path))
    summary = evaluator.error_analysis(loader, data, split="dev")

    out_path = tmp_path / "dev_errors.jsonl"
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(data)

    rec = json.loads(lines[0])
    for key in ("sample_id", "raw_text", "tokens", "gold_BIOES", "pred_BIOES",
                "gold_entities", "pred_entities", "start_gold", "start_pred",
                "end_gold", "end_pred", "error_categories"):
        assert key in rec

    assert (tmp_path / "dev_error_summary.json").exists()
    assert "focus_report" in summary
    assert "per_loc_len_ge2" in summary["focus_report"]
    assert "loc_admin_suffix" in summary["focus_report"]
