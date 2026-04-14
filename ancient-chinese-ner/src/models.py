"""
models.py — 2 NER Models:
  1. GuwenBertCRF    (GuwenBERT + CRF)
  2. GuwenBertLinear (GuwenBERT + Linear, no CRF)
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF
from src.utils import get_logger

logger = get_logger(__name__)

MODEL_BACKBONE = {
    "guwenbert_crf":    "ethanyt/guwenbert-base",
    "guwenbert_linear": "ethanyt/guwenbert-base",
}


# ── BACKBONE FORWARD HELPER ───────────────────────────────────────────────────
def _bert_forward(bert, input_ids, attention_mask, token_type_ids=None):
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None and getattr(bert.config, "type_vocab_size", 1) > 1:
        kwargs["token_type_ids"] = token_type_ids
    return bert(**kwargs).last_hidden_state


# ── SHARED CRF HELPER ─────────────────────────────────────────────────────────
def crf_step(crf, logits, attention_mask, labels=None):
    mask = attention_mask.bool()
    mask[:, 0] = True

    if labels is not None:
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0
        loss = -crf(logits, crf_labels, mask=mask, reduction="mean")
        return loss, logits

    return crf.decode(logits, mask=mask)


# ── MODEL 1: GuwenBERT + CRF ──────────────────────────────────────────────────
class GuwenBertCRF(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden    = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, num_labels)
        self.crf  = CRF(num_labels, batch_first=True)
        logger.info(f"GuwenBertCRF | backbone={backbone} | hidden={hidden} | labels={num_labels}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        seq    = self.drop(_bert_forward(self.bert, input_ids, attention_mask, token_type_ids))
        logits = self.fc(seq)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── MODEL 2: GuwenBERT + Linear (no CRF) ─────────────────────────────────────
class GuwenBertLinear(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(backbone)
        hidden       = self.bert.config.hidden_size
        self.drop    = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        logger.info(f"GuwenBertLinear | backbone={backbone} | hidden={hidden} | labels={num_labels}")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        seq    = self.drop(_bert_forward(self.bert, input_ids, attention_mask, token_type_ids))
        logits = self.fc(seq)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits

        # Return list-of-lists to match CRF decode format used by Evaluator/Trainer
        return logits.argmax(dim=-1).tolist()


# ── FACTORY ───────────────────────────────────────────────────────────────────
def build_model(cfg) -> nn.Module:
    """Build model từ OmegaConf config."""
    method   = cfg.model.method
    backbone = cfg.model.backbone or MODEL_BACKBONE.get(method)
    if backbone is None:
        raise ValueError(f"Unknown method: {method}. Valid: {list(MODEL_BACKBONE.keys())}")
    n_labels = cfg._num_labels

    if method == "guwenbert_crf":
        model = GuwenBertCRF(backbone, n_labels, cfg.model.dropout)

    elif method == "guwenbert_linear":
        model = GuwenBertLinear(backbone, n_labels, cfg.model.dropout)

    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Transfer learning: load BERT weights từ checkpoint ────────────────────
    pretrained_ckpt = getattr(cfg.model, "pretrained_ckpt", None)
    if pretrained_ckpt:
        logger.info(f"Loading pretrained BERT weights from: {pretrained_ckpt}")
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        bert_weights = {
            k[len("bert."):]: v
            for k, v in ckpt.items()
            if k.startswith("bert.")
        }
        missing, unexpected = model.bert.load_state_dict(bert_weights, strict=False)
        logger.info(f"  Loaded {len(bert_weights)} BERT weights")
        if missing:
            logger.warning(f"  Missing keys: {missing[:5]}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected[:5]}")

    return model
