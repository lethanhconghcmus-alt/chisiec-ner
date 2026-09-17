import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.bioes_utils import (
    bioes_to_spans,
    build_bioes_label_map,
    compute_boundary_pos_weight,
    repair_bioes_sequence,
)
from src.data_utils import NERDataset
from src.models import BertCRFBoundaryNER, binary_focal_loss_with_logits

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


# ── 10. BCE boundary loss không tính vào padding/CLS/SEP/-100 ──────────────
def test_boundary_loss_ignores_masked_positions():
    torch.manual_seed(0)
    label2id, _ = build_bioes_label_map()
    model = BertCRFBoundaryNER(TINY_BACKBONE, num_labels=len(label2id),
                                o_label_id=label2id["O"], boundary_loss_type="bce")

    # 1 batch, 5 positions: 2 valid (mask=True), 3 ignore (mask=False)
    logits = torch.tensor([[5.0, 5.0, -5.0, 999.0, -999.0]])  # ô 3,4 rác cực trị
    labels = torch.tensor([[1, 0, 1, 0, 1]])
    valid_mask = torch.tensor([[True, True, False, False, False]])

    loss_valid_only = model._boundary_loss(logits, labels, valid_mask, "start")

    # Tính tay loss chỉ trên 2 vị trí valid để so khớp
    manual = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[valid_mask], labels[valid_mask].float()
    )
    assert torch.allclose(loss_valid_only, manual, atol=1e-6)

    # Nếu vô tình include cả các vị trí rác (999/-999), loss sẽ khác hẳn
    full_mask = torch.ones_like(valid_mask, dtype=torch.bool)
    loss_with_junk = model._boundary_loss(logits, labels, full_mask, "start")
    assert not torch.allclose(loss_valid_only, loss_with_junk, atol=1e-3)


def test_boundary_loss_empty_valid_mask_returns_zero_grad_safe():
    label2id, _ = build_bioes_label_map()
    model = BertCRFBoundaryNER(TINY_BACKBONE, num_labels=len(label2id),
                                o_label_id=label2id["O"])
    logits = torch.tensor([[1.0, 2.0]], requires_grad=True)
    labels = torch.tensor([[0, 1]])
    mask = torch.zeros_like(labels, dtype=torch.bool)
    loss = model._boundary_loss(logits, labels, mask, "start")
    assert loss.item() == 0.0
    loss.backward()  # không được crash vì đứt graph


def test_focal_loss_matches_bce_when_gamma_zero_alpha_half():
    logits = torch.tensor([0.5, -1.2, 3.0, -0.3])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    focal = binary_focal_loss_with_logits(logits, targets, alpha=0.5, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal, 0.5 * bce, atol=1e-5)


def test_enable_boundary_auxiliary_false_total_loss_equals_crf_loss():
    torch.manual_seed(0)
    label2id, id2label = build_bioes_label_map()
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)
    tokens = ["太", "師", "陳"]
    labels = ["B-TITLE", "E-TITLE", "S-PER"]
    ds = NERDataset([(tokens, labels)], tokenizer, label2id, max_len=8, derive_boundary=True)
    item = ds[0]
    batch = {k: v.unsqueeze(0) for k, v in item.items()}

    model = BertCRFBoundaryNER(TINY_BACKBONE, num_labels=len(label2id),
                                o_label_id=label2id["O"],
                                enable_boundary_auxiliary=False)
    out = model(
        batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
        ner_labels=batch["labels"], start_labels=batch["start_labels"],
        end_labels=batch["end_labels"],
    )
    assert out.start_loss is None and out.end_loss is None
    assert torch.allclose(out.loss, out.crf_loss)


# ── 11. Overfit 20-50 samples: strict entity-F1 phải gần 1.0 ────────────────
def test_model_overfits_small_synthetic_dataset():
    """
    Dùng backbone tiny-random-bert (KHÔNG phải SikuBERT/GujiRoBERTa thật —
    quá lớn để test nhanh trên CPU không GPU) chỉ để xác minh wiring đúng
    (CRF + boundary heads học được trên vài chục câu tổng hợp lặp lại).
    Không phản ánh chất lượng thật trên DVSKTT — đó là việc train.py chạy
    trên Kaggle với backbone thật.
    """
    torch.manual_seed(0)
    label2id, id2label = build_bioes_label_map()
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)

    base_examples = [
        (list("太師陳守度至化州"),
         ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "B-LOC", "E-LOC"]),
        (list("黎時憲鄭根赴京畿"),
         ["B-PER", "I-PER", "E-PER", "B-PER", "E-PER", "O", "B-LOC", "E-LOC"]),
        (list("以黄義膠為督視"),
         ["O", "B-PER", "I-PER", "E-PER", "O", "B-TITLE", "E-TITLE"]),
    ]
    data = base_examples * 12  # 36 samples, lặp lại pattern để overfit nhanh

    train_loader_data = data
    ds = NERDataset(train_loader_data, tokenizer, label2id, max_len=16, derive_boundary=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=True)

    model = BertCRFBoundaryNER(
        TINY_BACKBONE, num_labels=len(label2id), o_label_id=label2id["O"],
        enable_boundary_auxiliary=True, boundary_weight=0.1,
        boundary_loss_type="bce",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    last_loss = None
    for epoch in range(40):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                ner_labels=batch["labels"], start_labels=batch["start_labels"],
                end_labels=batch["end_labels"],
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += out.loss.item()
        last_loss = epoch_loss / len(loader)

    # ── Đo strict entity-F1 trên chính train set (overfit check) ──────────
    model.eval()
    eval_loader = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=False)
    tp = fp = fn = 0
    debug_samples = []
    with torch.no_grad():
        for batch in eval_loader:
            out = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            for pred_seq, label_seq in zip(out.decoded_tags, batch["labels"]):
                gold_tags = [id2label[l.item()] for l in label_seq if l.item() != -100]
                # decoded_tags từ crf.decode: length = số vị trí valid theo mask
                # (mask[:,0]=True luôn tính CLS) -> bỏ phần tử đầu (CLS) để so
                # khớp với gold_tags (chỉ gồm token thật).
                pred_tags_full = [id2label[p] for p in pred_seq]
                pred_tags = pred_tags_full[1:1 + len(gold_tags)]

                pred_tags_repaired, _ = repair_bioes_sequence(pred_tags)
                gold_tags_repaired, _ = repair_bioes_sequence(gold_tags)
                gold_spans = set(bioes_to_spans(gold_tags_repaired))
                pred_spans = set(bioes_to_spans(pred_tags_repaired))

                tp += len(gold_spans & pred_spans)
                fp += len(pred_spans - gold_spans)
                fn += len(gold_spans - pred_spans)
                debug_samples.append((gold_tags, pred_tags))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    if f1 < 0.9:
        print(f"\n[OVERFIT TEST DEBUG] final_train_loss={last_loss:.4f} f1={f1:.4f} "
              f"P={precision:.4f} R={recall:.4f} tp={tp} fp={fp} fn={fn}")
        for gold, pred in debug_samples[:5]:
            print("  gold:", gold)
            print("  pred:", pred)

    assert f1 >= 0.9, (
        f"Overfit test thất bại: strict entity-F1={f1:.4f} (< 0.9) trên chính "
        f"train set sau 40 epoch — nghi ngờ lỗi wiring (label alignment, CRF "
        f"mask, boundary mask). last_train_loss={last_loss:.4f}"
    )
