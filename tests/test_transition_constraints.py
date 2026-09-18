import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchcrf import CRF

from src.bioes_utils import (
    apply_hard_transition_constraints,
    bioes_legal_end,
    bioes_legal_start,
    bioes_legal_transition,
    build_bioes_label_map,
    build_bioes_transition_masks,
)

# Bảng "Không được có" trong yêu cầu — mọi cặp này PHẢI bị coi là illegal.
SPEC_ILLEGAL_PAIRS = [
    ("O", "I-PER"), ("O", "E-PER"), ("O", "I-LOC"), ("O", "E-LOC"),
    ("B-PER", "O"), ("B-PER", "B-LOC"), ("B-PER", "I-LOC"), ("B-PER", "E-LOC"),
    ("I-PER", "O"), ("I-PER", "B-LOC"), ("I-PER", "I-LOC"), ("I-PER", "E-LOC"),
    ("E-PER", "I-LOC"), ("E-PER", "E-LOC"),
    ("S-PER", "I-LOC"), ("S-PER", "E-LOC"),
]

# Bảng "hợp lệ" trong yêu cầu
SPEC_LEGAL_PAIRS = [
    ("O", "O"), ("O", "B-PER"), ("O", "S-PER"),
    ("B-PER", "I-PER"), ("B-PER", "E-PER"),
    ("I-PER", "I-PER"), ("I-PER", "E-PER"),
    ("E-PER", "O"), ("E-PER", "B-LOC"), ("E-PER", "S-LOC"),
    ("S-PER", "O"), ("S-PER", "B-LOC"), ("S-PER", "S-LOC"),
]


def test_bioes_legal_transition_matches_spec_table():
    for prev, nxt in SPEC_ILLEGAL_PAIRS:
        assert bioes_legal_transition(prev, nxt) is False, f"{prev} -> {nxt} should be ILLEGAL"
    for prev, nxt in SPEC_LEGAL_PAIRS:
        assert bioes_legal_transition(prev, nxt) is True, f"{prev} -> {nxt} should be LEGAL"


def test_bioes_legal_start_end():
    assert bioes_legal_start("O") and bioes_legal_start("B-PER") and bioes_legal_start("S-PER")
    assert not bioes_legal_start("I-PER") and not bioes_legal_start("E-PER")
    assert bioes_legal_end("O") and bioes_legal_end("E-PER") and bioes_legal_end("S-PER")
    assert not bioes_legal_end("B-PER") and not bioes_legal_end("I-PER")


def test_build_bioes_transition_masks_shape_and_symmetry_with_label_map():
    label2id, id2label = build_bioes_label_map()
    illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)
    n = len(label2id)
    assert illegal_t.shape == (n, n)
    assert illegal_s.shape == (n,)
    assert illegal_e.shape == (n,)

    o_id = label2id["O"]
    bper_id = label2id["B-PER"]
    iper_id = label2id["I-PER"]
    eper_id = label2id["E-PER"]
    bloc_id = label2id["B-LOC"]

    assert illegal_t[o_id, iper_id].item() is True or bool(illegal_t[o_id, iper_id])
    assert not bool(illegal_t[o_id, bper_id])          # O -> B-PER hợp lệ
    assert bool(illegal_t[bper_id, o_id])               # B-PER -> O bất hợp lệ
    assert not bool(illegal_t[bper_id, iper_id])         # B-PER -> I-PER hợp lệ
    assert bool(illegal_t[bper_id, bloc_id])             # B-PER -> B-LOC (khác type) bất hợp lệ
    assert not bool(illegal_t[eper_id, bloc_id])         # E-PER -> B-LOC hợp lệ (đóng rồi mở mới)

    assert bool(illegal_s[iper_id])   # không được bắt đầu bằng I-*
    assert not bool(illegal_s[o_id])
    assert bool(illegal_e[bper_id])   # không được kết thúc bằng B-* (chưa đóng)
    assert not bool(illegal_e[eper_id])


def test_apply_hard_transition_constraints_sets_penalty_and_keeps_legal_learnable():
    label2id, _ = build_bioes_label_map()
    n = len(label2id)
    torch.manual_seed(0)
    crf = CRF(n, batch_first=True)
    illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)

    before_legal_val = crf.transitions[label2id["O"], label2id["B-PER"]].item()
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e, penalty=-100000.0)

    # Illegal entries đều bị ghi đè thành penalty
    assert torch.all(crf.transitions[illegal_t] == -100000.0)
    assert torch.all(crf.start_transitions[illegal_s] == -100000.0)
    assert torch.all(crf.end_transitions[illegal_e] == -100000.0)

    # Legal entries KHÔNG bị đụng vào (giữ nguyên giá trị random init ban đầu)
    after_legal_val = crf.transitions[label2id["O"], label2id["B-PER"]].item()
    assert before_legal_val == after_legal_val


def test_viterbi_decode_never_picks_illegal_transition_even_with_adversarial_emissions():
    """Cố tình cho emission score CỰC LỚN ủng hộ 1 chuỗi invalid (O rồi
    I-PER) để kiểm tra Viterbi decode vẫn KHÔNG chọn nó sau khi constrain."""
    label2id, id2label = build_bioes_label_map()
    n = len(label2id)
    crf = CRF(n, batch_first=True)
    illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e)

    seq_len = 3
    emissions = torch.zeros(1, seq_len, n)
    emissions[:, :, label2id["O"]] = 1.0
    # Ép cực mạnh vị trí 1 chọn I-PER (đứng sau O ở vị trí 0) -- đúng ra là
    # illegal (O -> I-PER), phải bị chặn dù emission áp đảo mọi lựa chọn khác.
    emissions[:, 1, label2id["I-PER"]] = 1e6
    mask = torch.ones(1, seq_len, dtype=torch.bool)

    decoded = crf.decode(emissions, mask=mask)[0]
    tags = [id2label[i] for i in decoded]
    # Constraint không cấm I-PER xuất hiện ở vị trí 1 -- nó cấm chuỗi ĐI ĐẾN
    # đó một cách bất hợp lệ. Viterbi phải tìm đường LEGAL khác vẫn hưởng
    # được emission khổng lồ đó (ở đây: B-PER I-PER E-PER) thay vì emit
    # thẳng O->I-PER. Assertion đúng là: mọi transition liền kề trong chuỗi
    # decode ra đều hợp lệ theo state machine BIOES.
    for prev, nxt in zip(tags, tags[1:]):
        assert bioes_legal_transition(prev, nxt), f"Illegal transition {prev} -> {nxt} trong {tags}"
