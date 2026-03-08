"""
models.py — 3 NER Models:
  1. BertCRF          (GuwenBERT + CRF)
  2. RobertaBiLSTMCRF (RoBERTa + BiLSTM + CRF)
  3. RobertaKANCRF    (RoBERTa + KAN + CRF)
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF
from src.utils import get_logger

logger = get_logger(__name__)

MODEL_BACKBONE = {
    "guwenbert_crf":       "ethanyt/guwenbert-base",
    "roberta_bilstm_crf":  "hfl/chinese-roberta-wwm-ext",
    "roberta_kan_crf":     "hfl/chinese-roberta-wwm-ext",
}


# ── BACKBONE FORWARD HELPER ───────────────────────────────────────────────
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
class BertCRF(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden    = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, num_labels)
        self.crf  = CRF(num_labels, batch_first=True)
        logger.info(f"BertCRF | backbone={backbone} | hidden={hidden} | labels={num_labels}")

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        seq    = self.drop(
            _bert_forward(self.bert, input_ids, attention_mask, token_type_ids)
        )
        logits = self.fc(seq)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── MODEL 2: RoBERTa + BiLSTM + CRF ──────────────────────────────────────────
class RobertaBiLSTMCRF(nn.Module):
    def __init__(self, backbone: str, num_labels: int, lstm_hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden    = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            hidden, lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc  = nn.Linear(lstm_hidden * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        logger.info(
            f"RobertaBiLSTMCRF | backbone={backbone} | "
            f"lstm_hidden={lstm_hidden}x2 | layers={num_layers} | labels={num_labels}"
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        seq       = self.drop(
            _bert_forward(self.bert, input_ids, attention_mask, token_type_ids)
        )
        lstm_o, _ = self.lstm(seq)
        logits    = self.fc(lstm_o)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── MODEL 3: RoBERTa + KAN + CRF ─────────────────────────────────────────────
class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 8):
        super().__init__()
        self.register_buffer("knots", torch.linspace(0, 1, num_knots))
        self.log_bandwidth = nn.Parameter(torch.zeros(1))
        self.coeff         = nn.Parameter(torch.randn(in_dim, num_knots, out_dim) * 0.02)
        self.residual      = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bw     = torch.exp(self.log_bandwidth)
        x_norm = torch.sigmoid(x)
        diff   = x_norm.unsqueeze(-1) - self.knots
        basis  = torch.exp(-bw * diff ** 2)
        kan_out = torch.einsum("blij,ijo->blo", basis, self.coeff)
        return kan_out + self.residual(x)


class RobertaKANCRF(nn.Module):
    def __init__(self, backbone: str, num_labels: int, kan_hidden: int = 512,
                 kan_knots: int = 8, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden    = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.kan  = KANLayer(hidden, kan_hidden, num_knots=kan_knots)
        self.norm = nn.LayerNorm(kan_hidden)
        self.fc   = nn.Linear(kan_hidden, num_labels)
        self.crf  = CRF(num_labels, batch_first=True)
        logger.info(
            f"RobertaKANCRF | backbone={backbone} | "
            f"kan_hidden={kan_hidden} | knots={kan_knots} | labels={num_labels}"
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        seq    = self.drop(
            _bert_forward(self.bert, input_ids, attention_mask, token_type_ids)
        )
        kan_o  = self.norm(self.kan(seq))
        logits = self.fc(kan_o)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── FACTORY ───────────────────────────────────────────────────────────────────
def build_model(cfg) -> nn.Module:
    """Build model từ OmegaConf config."""
    method   = cfg.model.method
    backbone = cfg.model.backbone or MODEL_BACKBONE[method]
    n_labels = cfg._num_labels

    if method == "guwenbert_crf":
        model = BertCRF(backbone, n_labels, cfg.model.dropout)

    elif method == "roberta_bilstm_crf":
        model = RobertaBiLSTMCRF(
            backbone, n_labels,
            lstm_hidden=cfg.model.lstm_hidden,
            num_layers=cfg.model.lstm_layers,
            dropout=cfg.model.dropout,
        )

    elif method == "roberta_kan_crf":
        model = RobertaKANCRF(
            backbone, n_labels,
            kan_hidden=cfg.model.kan_hidden,
            kan_knots=cfg.model.kan_knots,
            dropout=cfg.model.dropout,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # ── TRANSFER LEARNING: load BERT weights từ checkpoint cũ ────────────────
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
