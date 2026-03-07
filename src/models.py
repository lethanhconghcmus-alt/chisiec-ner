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



# ── BACKBONE FORWARD HELPER ───────────────────────────────────────────────
def _bert_forward(bert, input_ids, attention_mask, token_type_ids=None):
    """Call HF backbone safely across BERT/RoBERTa style models.
    - Some backbones ignore/ don't have token_type_ids (type_vocab_size==1).
    - Use keyword args to avoid signature mismatches.
    """
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None and getattr(bert.config, "type_vocab_size", 1) > 1:
        kwargs["token_type_ids"] = token_type_ids
    return bert(**kwargs).last_hidden_state


# ── SHARED CRF HELPER ─────────────────────────────────────────────────────────
def crf_step(
    crf: CRF,
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor | None = None,
):
    """
    Training  → returns (loss, logits)
    Inference → returns list of decoded tag sequences

    Notes:
    - torchcrf often expects float32; in fp16 training, cast logits to float32
      before CRF computation to avoid stalls / NaNs.
    - We mask out padding tokens via attention_mask, and we also ignore subword
      positions labeled as -100 by masking them out of CRF loss.
    """
    # torchcrf is safer in fp32 even under autocast
    logits = logits.float()
    base_mask = attention_mask.bool()

    if labels is not None:
        # Replace ignore index (-100) with a valid tag id (e.g. 0),
        # and mask those positions out of the CRF objective.
        crf_labels = labels.clone()
        ignore_mask = (crf_labels == -100)
        crf_labels[ignore_mask] = 0

        valid_mask = base_mask & (~ignore_mask)
        loss = -crf(logits, crf_labels, mask=valid_mask, reduction="mean")
        return loss, logits

    return crf.decode(logits, mask=base_mask)

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
            _bert_forward(self.bert, input_ids, attention_mask, token_type_ids)
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
            _bert_forward(self.bert, input_ids, attention_mask, token_type_ids)
        )
        lstm_o, _ = self.lstm(seq)
        logits    = self.fc(lstm_o)
        return crf_step(self.crf, logits, attention_mask, labels)


# ── MODEL 3: RoBERTa + KAN + CRF ─────────────────────────────────────────────
class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer dùng Gaussian RBF basis functions.
    Thay thế projection tuyến tính bằng các hàm đơn biến có thể học được,
    giúp nắm bắt các pattern phi tuyến trong văn bản cổ điển.
    """

    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 8):
        super().__init__()
        # Cố định knot positions (không train), chỉ train coefficients
        self.register_buffer("knots", torch.linspace(0, 1, num_knots))
        self.log_bandwidth = nn.Parameter(torch.zeros(1))          # learnable width
        self.coeff         = nn.Parameter(
            torch.randn(in_dim, num_knots, out_dim) * 0.02
        )
        self.residual      = nn.Linear(in_dim, out_dim)            # skip connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, in_dim)
        bw     = torch.exp(self.log_bandwidth)
        x_norm = torch.sigmoid(x)                                  # normalize to [0,1]
        diff   = x_norm.unsqueeze(-1) - self.knots                 # (B, L, in, K)
        basis  = torch.exp(-bw * diff ** 2)                        # (B, L, in, K)
        # Aggregate over knots weighted by learned coefficients
        kan_out = torch.einsum("blij,ijo->blo", basis, self.coeff) # (B, L, out)
        return kan_out + self.residual(x)                          # skip connection


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
