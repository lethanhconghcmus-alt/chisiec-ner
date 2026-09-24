"""
test_constrained_decode.py — mục D: constrained BIO Viterbi decode cho
GuwenBertCRF (benchmark_v2). Dùng adversarial emissions (cùng kỹ thuật
tests/test_transition_constraints.py) để CHỨNG MINH raw/unconstrained
decode CÓ THỂ chọn transition bất hợp lệ khi emission đủ lớn, còn
constrained decode (build_constrained_crf + torchcrf.CRF.decode thật)
KHÔNG BAO GIỜ chọn, bất kể emission điều chỉnh thế nào.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchcrf import CRF

from src.bioes_utils import find_bio_violations
from src.constrained_decode import build_constrained_crf

LABEL2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
N = len(LABEL2ID)


def _make_crf_with_transitions(transitions, start=None, end=None):
    torch.manual_seed(0)
    crf = CRF(N, batch_first=True)
    with torch.no_grad():
        crf.transitions.copy_(transitions)
        if start is not None:
            crf.start_transitions.copy_(start)
        if end is not None:
            crf.end_transitions.copy_(end)
    return crf


def _decode_ids_to_tags(decoded):
    return [ID2LABEL[i] for i in decoded[0]]


def test_orientation_transitions_i_j_is_from_i_to_j():
    """Xac nhan huong ma tran: transitions[i, j] = diem di TU tag i SANG tag
    j (dung huong nhu apply_hard_transition_constraints/torchcrf gia dinh)."""
    transitions = torch.zeros(N, N)
    transitions[LABEL2ID["B-PER"], LABEL2ID["I-PER"]] = 100.0  # B-PER -> I-PER duoc uu tien manh
    crf = _make_crf_with_transitions(transitions)
    emissions = torch.zeros(1, 2, N)
    emissions[0, 0, LABEL2ID["B-PER"]] = 10.0
    emissions[0, 1, LABEL2ID["I-PER"]] = 10.0
    mask = torch.ones(1, 2, dtype=torch.bool)
    decoded = crf.decode(emissions, mask=mask)
    assert _decode_ids_to_tags(decoded) == ["B-PER", "I-PER"]


def test_illegal_O_to_I_cannot_be_output_when_constrained():
    # emission adversarial: O roi I-PER duoc "thuong" cuc lon
    transitions = torch.zeros(N, N)
    transitions[LABEL2ID["O"], LABEL2ID["I-PER"]] = 1000.0
    crf = _make_crf_with_transitions(transitions)
    emissions = torch.zeros(1, 2, N)
    emissions[0, 0, LABEL2ID["O"]] = 100.0
    emissions[0, 1, LABEL2ID["I-PER"]] = 100.0
    mask = torch.ones(1, 2, dtype=torch.bool)

    raw = _decode_ids_to_tags(crf.decode(emissions, mask=mask))
    assert raw == ["O", "I-PER"], "setup phai chung minh RAW decode se chon duong bat hop le"

    constrained_crf = build_constrained_crf(crf, LABEL2ID, scheme="bio")
    constrained = _decode_ids_to_tags(constrained_crf.decode(emissions, mask=mask))
    assert constrained != ["O", "I-PER"]
    assert find_bio_violations(constrained) == []


def test_start_with_I_cannot_be_output_when_constrained():
    transitions = torch.zeros(N, N)
    start = torch.zeros(N)
    start[LABEL2ID["I-PER"]] = 1000.0
    crf = _make_crf_with_transitions(transitions, start=start)
    emissions = torch.zeros(1, 1, N)
    emissions[0, 0, LABEL2ID["I-PER"]] = 100.0
    mask = torch.ones(1, 1, dtype=torch.bool)

    raw = _decode_ids_to_tags(crf.decode(emissions, mask=mask))
    assert raw == ["I-PER"]

    constrained_crf = build_constrained_crf(crf, LABEL2ID, scheme="bio")
    constrained = _decode_ids_to_tags(constrained_crf.decode(emissions, mask=mask))
    assert constrained != ["I-PER"]
    assert find_bio_violations(constrained) == []


def test_mismatched_B_X_to_I_Y_cannot_be_output_when_constrained():
    transitions = torch.zeros(N, N)
    transitions[LABEL2ID["B-PER"], LABEL2ID["I-LOC"]] = 1000.0
    crf = _make_crf_with_transitions(transitions)
    emissions = torch.zeros(1, 2, N)
    emissions[0, 0, LABEL2ID["B-PER"]] = 100.0
    emissions[0, 1, LABEL2ID["I-LOC"]] = 100.0
    mask = torch.ones(1, 2, dtype=torch.bool)

    raw = _decode_ids_to_tags(crf.decode(emissions, mask=mask))
    assert raw == ["B-PER", "I-LOC"]

    constrained_crf = build_constrained_crf(crf, LABEL2ID, scheme="bio")
    constrained = _decode_ids_to_tags(constrained_crf.decode(emissions, mask=mask))
    assert constrained != ["B-PER", "I-LOC"]
    assert find_bio_violations(constrained) == []


def test_legal_B_X_to_I_X_path_remains_outputtable():
    transitions = torch.zeros(N, N)
    transitions[LABEL2ID["B-PER"], LABEL2ID["I-PER"]] = 100.0
    crf = _make_crf_with_transitions(transitions)
    emissions = torch.zeros(1, 2, N)
    emissions[0, 0, LABEL2ID["B-PER"]] = 10.0
    emissions[0, 1, LABEL2ID["I-PER"]] = 10.0
    mask = torch.ones(1, 2, dtype=torch.bool)

    constrained_crf = build_constrained_crf(crf, LABEL2ID, scheme="bio")
    constrained = _decode_ids_to_tags(constrained_crf.decode(emissions, mask=mask))
    assert constrained == ["B-PER", "I-PER"]


def test_special_and_pad_token_positions_excluded_before_violation_check():
    """Mo phong dung quy uoc thuc te cua pipeline: vi tri CLS/pad co gold
    label = -100, PHAI bi loai truoc khi dua vao find_bio_violations (giong
    _run_inference trong evaluator_extended.py). O day chi kiem tra ham
    filter dung, khong phu thuoc model that."""
    decoded_tags = ["B-PER", "I-PER", "O", "I-LOC"]  # vi tri 3 la "rac" tu CLS/pad
    gold_labels_raw = [1, -100, 0, -100]  # -100 = CLS/SEP/pad/subword tiep theo
    mask_raw = [1, 1, 1, 0]  # pad co mask=0

    filtered = [t for t, l, m in zip(decoded_tags, gold_labels_raw, mask_raw) if m == 1 and l != -100]
    assert filtered == ["B-PER", "O"]
    assert find_bio_violations(filtered) == []


def test_constrained_decode_zero_violations_on_random_adversarial_batch():
    """Fuzz nhe: transitions/emissions ngau nhien lon (de de sinh duong bat
    hop le neu khong constrain) -- constrained decode PHAI luon cho 0
    violation tren nhieu seed."""
    for seed in range(5):
        torch.manual_seed(seed)
        transitions = torch.randn(N, N) * 50
        start = torch.randn(N) * 50
        end = torch.randn(N) * 50
        crf = _make_crf_with_transitions(transitions, start=start, end=end)
        constrained_crf = build_constrained_crf(crf, LABEL2ID, scheme="bio")
        emissions = torch.randn(2, 6, N) * 50
        mask = torch.ones(2, 6, dtype=torch.bool)
        decoded = constrained_crf.decode(emissions, mask=mask)
        for seq in decoded:
            tags = [ID2LABEL[i] for i in seq]
            assert find_bio_violations(tags) == [], f"seed={seed} tags={tags}"
