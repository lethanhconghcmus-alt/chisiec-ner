"""
models.py — NER Models:
  1. GuwenBertCRF          (backbone + CRF, scheme BIO — baseline gốc)
  2. GuwenBertLinear       (backbone + Linear, no CRF)
  3. BertCRFBoundaryNER    (backbone + CRF (BIOES) + start/end boundary heads,
                            multi-task auxiliary training — xem mục C trong
                            yêu cầu). KHÔNG thay đổi hành vi 2 model trên.

Tên các method (guwenbert_crf, guwenbert_linear, bert_crf_boundary) là tên
KIẾN TRÚC, không phải tên backbone — backbone thật (GuwenBERT/SikuBERT/
GujiRoBERTa/...) truyền qua cfg.model.backbone (quy ước sẵn có trong các
kaggle kernel của repo này, ví dụ model.backbone=SIKU-BERT/sikubert).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel
from torchcrf import CRF
from src.utils import get_logger
from src.bioes_utils import apply_hard_transition_constraints, build_bioes_transition_masks

logger = get_logger(__name__)

MODEL_BACKBONE = {
    "guwenbert_crf":    "ethanyt/guwenbert-base",
    "guwenbert_linear": "ethanyt/guwenbert-base",
    # bert_crf_boundary: KHÔNG có default — backbone bắt buộc truyền qua
    # cfg.model.backbone (SikuBERT/GujiRoBERTa/GuwenBERT đều hợp lệ).
}


# ── BACKBONE FORWARD HELPER ───────────────────────────────────────────────────
def _bert_forward(bert, input_ids, attention_mask, token_type_ids=None):
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None and getattr(bert.config, "type_vocab_size", 1) > 1:
        kwargs["token_type_ids"] = token_type_ids
    return bert(**kwargs).last_hidden_state


# ── SHARED CRF HELPER ─────────────────────────────────────────────────────────
def crf_step(crf, logits, attention_mask, labels=None, o_label_id: int = 0):
    """
    o_label_id: id nhãn dùng thay cho -100 (CLS/SEP/pad/subword-tiếp-theo)
    khi build crf_labels. Mặc định=0 để giữ NGUYÊN hành vi cũ (GuwenBertCRF/
    GuwenBertLinear gọi hàm này không truyền tham số, id2label[0] có thể
    không phải "O" do build_label_map() sort alphabetically — đây là hành vi
    gốc của baseline, KHÔNG sửa để tránh đổi kết quả reproducibility của các
    checkpoint/kernel cũ). BertCRFBoundaryNER (scheme BIOES, label map cố
    định từ build_bioes_label_map — xem bioes_utils.py) truyền o_label_id=
    label2id["O"] tường minh để các vị trí bị ignore không nhiễu vào loss
    bằng một nhãn entity ngẫu nhiên.

    mask[:, 0] = True: bắt buộc bởi torchcrf (timestep đầu luôn phải valid,
    thư viện không hỗ trợ mask có "lỗ" ở giữa hai vị trí True) — đây là
    PHƯƠNG ÁN 1 trong yêu cầu (giữ CLS hợp lệ trong CRF mask, loại nó ra lúc
    decode/evaluate thay vì lúc build mask). Token thật sự bị ignore (CLS/
    SEP/pad/subword thứ 2+) đã có label=-100 nên không đóng góp gradient sai
    ở BCE/eval; ở CRF loss, việc gán chúng nhãn O không tạo tín hiệu học sai
    lệch đáng kể vì đây là non-boundary, non-entity token cực kỳ phổ biến.
    """
    mask = attention_mask.bool()
    mask[:, 0] = True

    if labels is not None:
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = o_label_id
        loss = -crf(logits, crf_labels, mask=mask, reduction="mean")
        return loss, logits

    return crf.decode(logits, mask=mask)


# ── MODEL 1: GuwenBERT + CRF ──────────────────────────────────────────────────
class GuwenBertCRF(nn.Module):
    def __init__(self, backbone: str, num_labels: int, dropout: float = 0.1,
                 gaz_vocab_size: int = 0, gaz_dim: int = 16, gaz_input_dim: int = 0):
        """gaz_vocab_size > 0: gazetteer categorical cũ (1 nhãn/vị trí, nn.Embedding).
        gaz_input_dim > 0: gazetteer multi-hot mới (nhiều cờ độc lập/vị trí,
        nn.Linear) -- ưu tiên dùng khi cả hai cùng > 0."""
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        hidden    = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)

        self.use_gaz_multihot = gaz_input_dim > 0
        self.use_gaz_categorical = (not self.use_gaz_multihot) and gaz_vocab_size > 0
        self.use_gaz = self.use_gaz_multihot or self.use_gaz_categorical

        if self.use_gaz_multihot:
            self.gaz_proj = nn.Linear(gaz_input_dim, gaz_dim)
            fc_in = hidden + gaz_dim
        elif self.use_gaz_categorical:
            self.gaz_embed = nn.Embedding(gaz_vocab_size, gaz_dim, padding_idx=0)
            fc_in = hidden + gaz_dim
        else:
            fc_in = hidden

        self.fc  = nn.Linear(fc_in, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        gaz_desc = ""
        if self.use_gaz_multihot:
            gaz_desc = f" | gaz_multihot_dim={gaz_input_dim} | gaz_dim={gaz_dim}"
        elif self.use_gaz_categorical:
            gaz_desc = f" | gaz_vocab={gaz_vocab_size} | gaz_dim={gaz_dim}"
        logger.info(
            f"GuwenBertCRF | backbone={backbone} | hidden={hidden} | labels={num_labels}{gaz_desc}"
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, gaz_ids=None):
        seq = self.drop(_bert_forward(self.bert, input_ids, attention_mask, token_type_ids))
        if self.use_gaz_multihot:
            if gaz_ids is None:
                gaz_ids = torch.zeros(*seq.shape[:2], self.gaz_proj.in_features,
                                       dtype=torch.float, device=seq.device)
            seq = torch.cat([seq, self.gaz_proj(gaz_ids)], dim=-1)
        elif self.use_gaz_categorical:
            if gaz_ids is None:
                gaz_ids = torch.zeros(seq.shape[:2], dtype=torch.long, device=seq.device)
            seq = torch.cat([seq, self.gaz_embed(gaz_ids)], dim=-1)
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

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, gaz_ids=None):
        seq    = self.drop(_bert_forward(self.bert, input_ids, attention_mask, token_type_ids))
        logits = self.fc(seq)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits

        # Return list-of-lists to match CRF decode format used by Evaluator/Trainer
        return logits.argmax(dim=-1).tolist()


# ── FOCAL LOSS (binary, from logits, numerically stable) ─────────────────────
def binary_focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor,
    alpha: float = 0.25, gamma: float = 2.0,
) -> torch.Tensor:
    """logits/targets: 1D tensor đã lọc theo valid_ner_mask (không còn -100/pad)."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    modulating = (1.0 - p_t).clamp(min=0.0) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * modulating * ce).mean()


@dataclass
class BoundaryNEROutput:
    loss: Optional[torch.Tensor] = None
    crf_loss: Optional[torch.Tensor] = None
    start_loss: Optional[torch.Tensor] = None
    end_loss: Optional[torch.Tensor] = None
    emissions: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    decoded_tags: Optional[list] = None


# ── MODEL 3: backbone + CRF (BIOES) + start/end boundary heads ───────────────
class BertCRFBoundaryNER(nn.Module):
    """
    encoder (SikuBERT/GujiRoBERTa/GuwenBERT/...) -> shared hidden state ->
        ├── ner_classifier -> CRF -> BIOES tags
        ├── start_classifier -> binary start probability
        └── end_classifier   -> binary end probability

    Boundary heads là auxiliary training signal (multi-task). total_loss =
    crf_loss + boundary_weight * (start_loss + end_loss) khi
    enable_boundary_auxiliary=True; nếu False, total_loss = crf_loss và
    boundary loss không được tính (model vẫn trả start/end logits nếu
    labels có sẵn, nhưng không dùng để backprop — xem forward()).

    V1 (mục F trong yêu cầu): inference NER cuối cùng CHỈ dùng CRF decode
    thuần; start/end logits KHÔNG được dùng để post-process CRF output ở
    class này — post-processing bằng boundary score (nếu cần) nên là bước
    riêng ở evaluator/inference script, không phải trách nhiệm của model.
    """

    def __init__(
        self,
        backbone: str,
        num_labels: int,
        o_label_id: int = 0,
        dropout: float = 0.1,
        enable_boundary_auxiliary: bool = True,
        boundary_weight: float = 0.1,
        boundary_loss_type: str = "weighted_bce",
        start_pos_weight: Optional[float] = None,
        end_pos_weight: Optional[float] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label2id: Optional[dict] = None,
        constrain_bioes_transitions: bool = True,
        transition_constraint_penalty: float = -100000.0,
    ):
        super().__init__()
        if boundary_loss_type not in ("bce", "weighted_bce", "focal"):
            raise ValueError(f"Unknown boundary_loss_type: {boundary_loss_type!r}")

        self.bert = AutoModel.from_pretrained(backbone)
        hidden = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)

        self.ner_classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.start_classifier = nn.Linear(hidden, 1)
        self.end_classifier = nn.Linear(hidden, 1)

        self.o_label_id = o_label_id
        self.enable_boundary_auxiliary = enable_boundary_auxiliary
        self.boundary_weight = boundary_weight
        self.boundary_loss_type = boundary_loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # pos_weight lưu dạng python float, chuyển tensor + .to(device) lúc
        # dùng (không register_buffer vì có thể là None và cần đơn giản).
        self.start_pos_weight = start_pos_weight
        self.end_pos_weight = end_pos_weight

        # ── Hard-constrained CRF transitions (BIOES state machine) ────────
        # torchcrf không tự enforce constraint -- xem docstring
        # src/bioes_utils.py:apply_hard_transition_constraints. Cần
        # label2id để biết nhãn nào ứng với index nào trong ma trận
        # transitions[num_labels, num_labels]. constrain_bioes_transitions=
        # False (hoặc label2id=None) -> tắt hẳn, giữ hành vi CRF gốc
        # (torchcrf tự học transition, có thể decode ra chuỗi invalid).
        self.constrain_bioes_transitions = constrain_bioes_transitions and (label2id is not None)
        if self.constrain_bioes_transitions:
            illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)
            self.register_buffer("_illegal_transition", illegal_t)
            self.register_buffer("_illegal_start", illegal_s)
            self.register_buffer("_illegal_end", illegal_e)
            self.transition_constraint_penalty = transition_constraint_penalty
            self.apply_transition_constraints()
        else:
            self._illegal_transition = None

        logger.info(
            f"BertCRFBoundaryNER | backbone={backbone} | hidden={hidden} | "
            f"labels={num_labels} | o_label_id={o_label_id} | "
            f"enable_boundary_auxiliary={enable_boundary_auxiliary} | "
            f"boundary_weight={boundary_weight} | boundary_loss_type={boundary_loss_type} | "
            f"constrain_bioes_transitions={self.constrain_bioes_transitions}"
        )

    def apply_transition_constraints(self):
        """Gọi lại sau MỖI optimizer.step() (xem trainer_boundary.py) để CRF
        transitions không bao giờ "trôi" khỏi ràng buộc BIOES hợp lệ, dù
        gradient của NLL loss có thể nhích nhẹ các ô bị cấm lên sau mỗi
        bước cập nhật. No-op nếu constrain_bioes_transitions=False."""
        if not self.constrain_bioes_transitions:
            return
        apply_hard_transition_constraints(
            self.crf, self._illegal_transition, self._illegal_start,
            self._illegal_end, penalty=self.transition_constraint_penalty,
        )

    def _boundary_loss(self, logits: torch.Tensor, target_labels: torch.Tensor,
                        valid_mask: torch.Tensor, kind: str) -> torch.Tensor:
        logits_flat = logits[valid_mask]
        targets_flat = target_labels[valid_mask].float()

        if logits_flat.numel() == 0:
            # Không có vị trí valid nào trong batch (không nên xảy ra với
            # câu thật, nhưng giữ an toàn để không crash) -> zero-loss có
            # gradient hợp lệ (nối qua logits để không đứt graph).
            return logits.sum() * 0.0

        if self.boundary_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits_flat, targets_flat)

        if self.boundary_loss_type == "weighted_bce":
            pw = self.start_pos_weight if kind == "start" else self.end_pos_weight
            pos_weight = None
            if pw is not None:
                pos_weight = torch.tensor(float(pw), device=logits_flat.device)
            return F.binary_cross_entropy_with_logits(
                logits_flat, targets_flat, pos_weight=pos_weight
            )

        # focal
        return binary_focal_loss_with_logits(
            logits_flat, targets_flat, alpha=self.focal_alpha, gamma=self.focal_gamma
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        ner_labels=None,
        start_labels=None,
        end_labels=None,
        boundary_weight: Optional[float] = None,
    ) -> BoundaryNEROutput:
        seq = self.drop(_bert_forward(self.bert, input_ids, attention_mask, token_type_ids))
        emissions = self.ner_classifier(seq)
        start_logits = self.start_classifier(seq).squeeze(-1)
        end_logits = self.end_classifier(seq).squeeze(-1)

        bw = self.boundary_weight if boundary_weight is None else boundary_weight

        crf_loss = None
        decoded_tags = None
        start_loss = end_loss = None
        total_loss = None

        if ner_labels is not None:
            crf_loss, _ = crf_step(
                self.crf, emissions, attention_mask, ner_labels, o_label_id=self.o_label_id
            )
            total_loss = crf_loss

            if self.enable_boundary_auxiliary and start_labels is not None and end_labels is not None:
                # valid_ner_mask (mục C): attention thật AND có nhãn NER hợp
                # lệ (loại CLS/SEP/pad/subword-tiếp-theo đã đánh -100).
                valid_ner_mask = attention_mask.bool() & ner_labels.ne(-100)
                start_loss = self._boundary_loss(start_logits, start_labels, valid_ner_mask, "start")
                end_loss = self._boundary_loss(end_logits, end_labels, valid_ner_mask, "end")
                total_loss = crf_loss + bw * (start_loss + end_loss)
        else:
            decoded_tags = crf_step(
                self.crf, emissions, attention_mask, None, o_label_id=self.o_label_id
            )

        return BoundaryNEROutput(
            loss=total_loss,
            crf_loss=crf_loss,
            start_loss=start_loss,
            end_loss=end_loss,
            emissions=emissions,
            start_logits=start_logits,
            end_logits=end_logits,
            decoded_tags=decoded_tags,
        )


# ── FACTORY ───────────────────────────────────────────────────────────────────
def build_model(cfg, label2id: Optional[dict] = None) -> nn.Module:
    """Build model từ OmegaConf config. label2id: CHỈ cần cho
    method=bert_crf_boundary (để bake hard transition constraint BIOES —
    xem BertCRFBoundaryNER); None ở mọi call site khác (guwenbert_crf/
    guwenbert_linear không dùng, giữ nguyên hành vi cũ)."""
    method   = cfg.model.method
    backbone = cfg.model.backbone or MODEL_BACKBONE.get(method)
    if backbone is None:
        raise ValueError(f"Unknown method: {method}. Valid: {list(MODEL_BACKBONE.keys())}")
    n_labels = cfg._num_labels

    if method == "guwenbert_crf":
        gaz_vocab_size = int(getattr(cfg.model, "gaz_vocab_size", 0) or 0)
        gaz_input_dim  = int(getattr(cfg.model, "gaz_input_dim", 0) or 0)
        gaz_dim        = int(getattr(cfg.model, "gaz_dim", 16) or 16)
        model = GuwenBertCRF(backbone, n_labels, cfg.model.dropout,
                              gaz_vocab_size=gaz_vocab_size, gaz_dim=gaz_dim,
                              gaz_input_dim=gaz_input_dim)

    elif method == "guwenbert_linear":
        model = GuwenBertLinear(backbone, n_labels, cfg.model.dropout)

    elif method == "bert_crf_boundary":
        if not cfg.model.backbone:
            raise ValueError(
                "method=bert_crf_boundary bắt buộc phải chỉ định cfg.model.backbone "
                "tường minh (ví dụ SIKU-BERT/sikubert hoặc hsc748NLP/GujiRoBERTa_jian)."
            )
        mcfg = cfg.model
        model = BertCRFBoundaryNER(
            backbone,
            n_labels,
            o_label_id=int(getattr(cfg, "_o_label_id", 0) or 0),
            dropout=mcfg.dropout,
            enable_boundary_auxiliary=bool(getattr(mcfg, "enable_boundary_auxiliary", True)),
            boundary_weight=float(getattr(mcfg, "boundary_weight", 0.1)),
            boundary_loss_type=str(getattr(mcfg, "boundary_loss_type", "weighted_bce")),
            start_pos_weight=getattr(cfg, "_start_pos_weight", None),
            end_pos_weight=getattr(cfg, "_end_pos_weight", None),
            focal_alpha=float(getattr(mcfg, "focal_alpha", 0.25)),
            focal_gamma=float(getattr(mcfg, "focal_gamma", 2.0)),
            label2id=label2id,
            constrain_bioes_transitions=bool(getattr(mcfg, "constrain_bioes_transitions", True)),
        )

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
