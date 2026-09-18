import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.bioes_utils import build_transition_masks
from src.data_utils import NERDataset, build_label_map
from src.models import BertCRFBoundaryNER

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


def test_m3_bio_boundary_pipeline_overfits_small_synthetic_dataset():
    """
    Xác nhận toàn bộ pipeline M3 (BIO + boundary heads, KHÔNG convert
    BIOES) hoạt động đúng end-to-end trước khi chạy trên Kaggle: dataset
    align đúng boundary_scheme='bio', model constrain đúng scheme 'bio'
    (KHÔNG áp nhầm luật BIOES 21-nhãn lên 11-nhãn BIO), và học được.
    """
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(TINY_BACKBONE)

    base_examples = [
        (list("太師陳守度至化州"),
         ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]),
        (list("黎時憲鄭根赴京畿"),
         ["B-PER", "I-PER", "I-PER", "B-PER", "I-PER", "O", "B-LOC", "I-LOC"]),
        (list("以黄義膠為督視"),
         ["O", "B-PER", "I-PER", "I-PER", "O", "B-TITLE", "I-TITLE"]),
    ]
    data = base_examples * 12

    label2id, id2label = build_label_map(data)
    assert len(label2id) == 7  # O + B/I x {PER,LOC,TITLE} co trong data nay (khong ORG/DTM)

    ds = NERDataset(data, tokenizer, label2id, max_len=16,
                     derive_boundary=True, boundary_scheme="bio")
    loader = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=True)

    model = BertCRFBoundaryNER(
        TINY_BACKBONE, num_labels=len(label2id), o_label_id=label2id["O"],
        label2id=label2id, label_scheme="bio", constrain_bioes_transitions=True,
        enable_boundary_auxiliary=True, boundary_weight=0.1, boundary_loss_type="bce",
    )
    assert model.label_scheme == "bio"
    assert model.constrain_bioes_transitions is True

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model.train()
    for _ in range(40):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"],
                        ner_labels=batch["labels"], start_labels=batch["start_labels"],
                        end_labels=batch["end_labels"])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.apply_transition_constraints()

    # ── Đo strict entity-F1 (BIO, dùng seqeval) trên chính train set ──────
    from seqeval.metrics import f1_score

    model.eval()
    eval_loader = torch.utils.data.DataLoader(ds, batch_size=6, shuffle=False)
    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in eval_loader:
            out = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            for pred_seq, label_seq in zip(out.decoded_tags, batch["labels"]):
                gold_tags = [id2label[l.item()] for l in label_seq if l.item() != -100]
                pred_tags_full = [id2label[p] for p in pred_seq]
                pred_tags = pred_tags_full[1: 1 + len(gold_tags)]
                all_gold.append(gold_tags)
                all_pred.append(pred_tags)

    f1 = f1_score(all_gold, all_pred, zero_division=0)
    assert f1 >= 0.9, f"M3 overfit test thất bại: F1={f1:.4f} (< 0.9)"


def test_build_transition_masks_bio_matches_11_label_space_from_real_label_map():
    data = [(["a"], ["O"]), (["a"], ["B-PER"]), (["a"], ["I-PER"])]
    label2id, _ = build_label_map(data)
    illegal_t, illegal_s, illegal_e = build_transition_masks(label2id, scheme="bio")
    assert illegal_t.shape == (len(label2id), len(label2id))
