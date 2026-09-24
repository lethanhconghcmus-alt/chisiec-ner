"""
constrained_decode.py — BIO-constrained Viterbi decode áp dụng CHỈ ở thời
điểm decode/evaluate (KHÔNG đụng training loss/optimizer step), dành cho
GuwenBertCRF (kiến trúc benchmark_v2: Linear + CRF + BIO, KHÔNG gazetteer/
boundary heads).

Cơ chế: deepcopy RIÊNG 3 tensor tham số nhỏ (transitions/start_transitions/
end_transitions) của `model.crf`, áp hard transition constraint (-inf) LÊN
BẢN SAO qua `src.bioes_utils.apply_hard_transition_constraints()` (100% cơ
chế ĐÃ CÓ SẴN và ĐÃ TEST cho M2/M3 — tests/test_transition_constraints.py),
KHÔNG đụng chạm `model.crf` gốc. Vì vậy:
- Training/loss/optimizer.step() hoàn toàn KHÔNG bị ảnh hưởng (không có
  chỗ nào trong module này chạy trong vòng lặp train, không có backward).
- Decode dùng nguyên `torchcrf.CRF.decode()` thật (cùng package, cùng
  orientation ma trận transitions[i,j] = score đi TỪ tag i SANG tag j —
  xem docstring `apply_hard_transition_constraints`) — không tự viết lại
  thuật toán Viterbi, giảm bề mặt lỗi.
"""

from __future__ import annotations

import copy

import torch

from src.bioes_utils import apply_hard_transition_constraints, build_transition_masks
from src.models import _bert_forward


def build_constrained_crf(crf, label2id: dict, scheme: str = "bio"):
    """Trả về 1 BẢN SAO của `crf` đã áp hard BIO transition constraint.
    KHÔNG sửa đổi `crf` gốc (deepcopy trước khi masked_fill_)."""
    illegal_transition, illegal_start, illegal_end = build_transition_masks(label2id, scheme=scheme)
    crf_copy = copy.deepcopy(crf)
    apply_hard_transition_constraints(crf_copy, illegal_transition, illegal_start, illegal_end)
    return crf_copy


@torch.no_grad()
def guwenbert_crf_logits(model, input_ids, attention_mask, token_type_ids=None):
    """Lặp lại CHÍNH XÁC phần pre-CRF của GuwenBertCRF.forward() (xem
    src/models.py) để lấy logits thô, KHÔNG gọi crf.decode() (unconstrained)
    như forward() mặc định. CHỈ hỗ trợ GuwenBertCRF KHÔNG gazetteer (đúng
    cấu hình benchmark_v2) -- nếu model có bật gazetteer, raise rõ ràng
    thay vì âm thầm bỏ qua feature đó."""
    if getattr(model, "use_gaz", False):
        raise NotImplementedError(
            "guwenbert_crf_logits() chua ho tro gazetteer -- benchmark_v2 KHONG dung "
            "gazetteer nen chua can; them nhanh gaz o day neu tai su dung cho case khac."
        )
    seq = model.drop(_bert_forward(model.bert, input_ids, attention_mask, token_type_ids))
    return model.fc(seq)


@torch.no_grad()
def constrained_decode_batch(model, constrained_crf, input_ids, attention_mask, token_type_ids=None):
    """Decode 1 batch bằng `constrained_crf` (đã áp hard transition mask)
    thay vì model.crf gốc. Dùng CHÍNH XÁC cách build mask giống crf_step()
    (mask[:, 0] = True — xem src/models.py:crf_step docstring, bắt buộc bởi
    torchcrf) để giữ nguyên hành vi vị trí [CLS] như pipeline gốc."""
    logits = guwenbert_crf_logits(model, input_ids, attention_mask, token_type_ids)
    mask = attention_mask.bool().clone()
    mask[:, 0] = True
    return constrained_crf.decode(logits, mask=mask)
