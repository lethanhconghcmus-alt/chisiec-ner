"""
models.py — 3 NER Models:
  1. BertCRF          (GuwenBERT + CRF)
  2. RobertaBiLSTMCRF (RoBERTa + BiLSTM + CRF)
  3. RobertaKANCRF    (RoBERTa + KAN + CRF)
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from TorchCRF import CRF

from src.utils import get_logger

logger = get_logger(__name__)

MODEL_BACKBONE = {
    "guwenbert_crf":       "ethanyt/guwenbert-base",
    "roberta_bilstm_crf":  "hfl/chinese-roberta-wwm-ext",
    "roberta_kan_crf":     "hfl/chinese-roberta-wwm-ext",
}


# ── SHARED CRF HELPER ─────────────────────────────────────────────────────────
def crf_step(
    crf: CRF,
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    labels = None,
):
    """
    Training  → returns (loss, logits)
    Inference → returns list of decoded tag sequences
    """
    mask = attention_mask.bool()
    if labels is not None:
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0
        valid_mask = mask
        loss = -crf(logits, crf_labels, mask=valid_mask, reduction="mean")
        return loss, logits
    return crf.decode(logits, mask=mask)


# ── MODEL 1: GuwenBERT + CRF ──────────────────────────────────────────────────
class BertCRF(nn.Module):
    """
    BERT/GuwenBERT encoder → Dropout → Linear → CRF
    Kiến trúc đơn giản, hiệu quả cao cho văn bản cổ Trung Quốc.
    """

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
            self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        )
        logits = self.fc(seq)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── MODEL 2: RoBERTa + BiLSTM + CRF ──────────────────────────────────────────
class RobertaBiLSTMCRF(nn.Module):
    """
    RoBERTa encoder → BiLSTM (capture long-range context) → CRF
    BiLSTM giúp mô hình nắm bắt phụ thuộc chuỗi tốt hơn BERT đơn thuần.
    """

    def __init__(
        self,
        backbone:    str,
        num_labels:  int,
        lstm_hidden: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ):
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
            self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        )
        lstm_o, _ = self.lstm(seq)
        logits    = self.fc(lstm_o)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── MODEL 3: RoBERTa + KAN + CRF ─────────────────────────────────────────────
class KANLayer(nn.Module):
    """
    Optimized KAN layer: project input xuống proj_dim trước khi apply RBF basis,
    giảm tensor size từ (B,L,768,K) xuống (B,L,proj_dim,K) → nhanh hơn ~6x.
    """

    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 8, proj_dim: int = 128):
        super().__init__()
        # 1) Project xuống proj_dim trước (linear, cheap)
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        # 2) KAN trên proj_dim (nhỏ hơn nhiều)
        self.register_buffer("knots", torch.linspace(0, 1, num_knots))
        self.log_bandwidth = nn.Parameter(torch.zeros(1))
        self.coeff = nn.Parameter(
            torch.randn(proj_dim, num_knots, out_dim) * 0.02
        )
        # 3) Skip connection từ in_dim gốc
        self.residual = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_dim)
        bw     = torch.exp(self.log_bandwidth)
        xp     = torch.sigmoid(self.proj(x))           # (B, L, proj_dim)
        diff   = xp.unsqueeze(-1) - self.knots          # (B, L, proj_dim, K)
        basis  = torch.exp(-bw * diff ** 2)             # (B, L, proj_dim, K)
        # einsum giờ nhỏ hơn: (B,L,128,8) x (128,8,out_dim)
        kan_out = torch.einsum("blpk,pko->blo", basis, self.coeff)  # (B, L, out_dim)
        return kan_out + self.residual(x)


class RobertaKANCRF(nn.Module):
    """
    RoBERTa encoder → KAN (non-linear feature transformation) → LayerNorm → CRF
    KAN học các biến đổi phi tuyến linh hoạt hơn Linear projection.
    """

    def __init__(
        self,
        backbone:   str,
        num_labels: int,
        kan_hidden: int   = 512,
        kan_knots:  int   = 8,
        dropout:    float = 0.1,
    ):
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
            self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        )
        kan_o  = self.norm(self.kan(seq))
        logits = self.fc(kan_o)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── FACTORY ───────────────────────────────────────────────────────────────────
def build_model(cfg) -> nn.Module:
    """Build model từ OmegaConf config."""
    method   = cfg.model.method
    backbone = cfg.model.backbone or MODEL_BACKBONE[method]
    n_labels = cfg._num_labels       # set by trainer after building label map

    if method == "guwenbert_crf":
        return BertCRF(backbone, n_labels, cfg.model.dropout)

    elif method == "roberta_bilstm_crf":
        return RobertaBiLSTMCRF(
            backbone, n_labels,
            lstm_hidden=cfg.model.lstm_hidden,
            num_layers=cfg.model.lstm_layers,
            dropout=cfg.model.dropout,
        )

    elif method == "roberta_kan_crf":
        return RobertaKANCRF(
            backbone, n_labels,
            kan_hidden=cfg.model.kan_hidden,
            kan_knots=cfg.model.kan_knots,
            dropout=cfg.model.dropout,
        )

    else:
        raise ValueError(f"Unknown method: {method}")
