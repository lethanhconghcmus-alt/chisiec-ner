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
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e)  # default penalty=-inf

    # Illegal entries đều bị ghi đè thành -inf (hard constraint thật, xem
    # docstring apply_hard_transition_constraints() lý do không dùng số âm
    # hữu hạn như -100000).
    assert torch.all(torch.isneginf(crf.transitions[illegal_t]))
    assert torch.all(torch.isneginf(crf.start_transitions[illegal_s]))
    assert torch.all(torch.isneginf(crf.end_transitions[illegal_e]))

    # Legal entries KHÔNG bị đụng vào (giữ nguyên giá trị random init ban đầu)
    after_legal_val = crf.transitions[label2id["O"], label2id["B-PER"]].item()
    assert before_legal_val == after_legal_val


def test_viterbi_never_starts_with_I_or_E_even_with_adversarial_emissions():
    """Ép token đầu tiên chọn I-PER (hoặc E-PER) bằng emission cực lớn --
    decoder PHẢI không bao giờ trả I-*/E-* ở vị trí đầu (chỉ O/B-*/S-* hợp
    lệ làm điểm bắt đầu). Kiểm tra riêng start_transitions, tách biệt khỏi
    transitions nội bộ đã test ở trên."""
    label2id, id2label = build_bioes_label_map()
    n = len(label2id)
    crf = CRF(n, batch_first=True)
    illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e)

    emissions = torch.zeros(1, 2, n)
    emissions[:, 0, label2id["I-PER"]] = 1e6   # ép cực mạnh I-PER ở vị trí 0
    emissions[:, 1, label2id["O"]] = 1.0
    mask = torch.ones(1, 2, dtype=torch.bool)

    tags = [id2label[i] for i in crf.decode(emissions, mask=mask)[0]]
    assert not tags[0].startswith("I-"), f"I-* xuất hiện ở vị trí đầu: {tags}"
    assert not tags[0].startswith("E-"), f"E-* xuất hiện ở vị trí đầu: {tags}"
    assert bioes_legal_start(tags[0])


def test_viterbi_never_ends_with_B_or_I_even_with_adversarial_emissions():
    """Ép token CUỐI chọn B-PER (hoặc I-PER) bằng emission cực lớn --
    decoder PHẢI không bao giờ trả B-*/I-* ở vị trí cuối (chỉ O/E-*/S-*
    hợp lệ làm điểm kết thúc). Kiểm tra riêng end_transitions."""
    label2id, id2label = build_bioes_label_map()
    n = len(label2id)
    crf = CRF(n, batch_first=True)
    illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e)

    emissions = torch.zeros(1, 2, n)
    emissions[:, 0, label2id["O"]] = 1.0
    emissions[:, 1, label2id["B-PER"]] = 1e6   # ép cực mạnh B-PER ở vị trí cuối
    mask = torch.ones(1, 2, dtype=torch.bool)

    tags = [id2label[i] for i in crf.decode(emissions, mask=mask)[0]]
    assert not tags[-1].startswith("B-"), f"B-* xuất hiện ở vị trí cuối: {tags}"
    assert not tags[-1].startswith("I-"), f"I-* xuất hiện ở vị trí cuối: {tags}"
    assert bioes_legal_end(tags[-1])


def test_viterbi_rejects_cross_type_B_then_E_even_with_adversarial_emissions():
    """Ép chuỗi 'B-PER rồi E-LOC' (khác type) bằng emission cực lớn cho cả
    2 vị trí -- decoder KHÔNG được trả đúng cặp bất hợp lệ đó nguyên văn."""
    label2id, id2label = build_bioes_label_map()
    n = len(label2id)
    crf = CRF(n, batch_first=True)
    illegal_t, illegal_s, illegal_e = build_bioes_transition_masks(label2id)
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e)

    emissions = torch.zeros(1, 2, n)
    emissions[:, 0, label2id["B-PER"]] = 1e6
    emissions[:, 1, label2id["E-LOC"]] = 1e6
    mask = torch.ones(1, 2, dtype=torch.bool)

    tags = [id2label[i] for i in crf.decode(emissions, mask=mask)[0]]
    assert tags != ["B-PER", "E-LOC"], f"Cặp bất hợp lệ B-PER->E-LOC vẫn được decode: {tags}"
    assert bioes_legal_transition(tags[0], tags[1])


def test_transitions_matrix_orientation_is_from_row_to_column():
    """Xác nhận đúng chiều torchcrf.CRF.transitions[i, j] = score CHUYỂN TỪ
    i SANG j (không phải ngược lại) -- mask illegal_transition[i,j] build
    trong build_bioes_transition_masks() phải khớp đúng chiều này, nếu
    không toàn bộ constraint sẽ áp nhầm cặp. Kiểm tra trực tiếp bằng cách
    set 1 giá trị transitions[i,j] cực đoan và xem nó ảnh hưởng đến việc
    "đến j sau i" hay "đến i sau j"."""
    label2id, id2label = build_bioes_label_map()
    n = len(label2id)
    crf = CRF(n, batch_first=True)
    o_id, bper_id = label2id["O"], label2id["B-PER"]

    with torch.no_grad():
        crf.transitions.zero_()
        crf.start_transitions.zero_()
        crf.end_transitions.zero_()
        # Nếu transitions[i, j] = "i -> j", set transitions[O, B-PER] rất
        # lớn phải làm decoder thích đi TỪ O ĐẾN B-PER (thứ tự O rồi B-PER),
        # không phải B-PER rồi O.
        crf.transitions[o_id, bper_id] = 1e6

    n_labels = n
    emissions = torch.zeros(1, 2, n_labels)  # emission trung lập, chỉ transition quyết định
    mask = torch.ones(1, 2, dtype=torch.bool)
    tags = [id2label[i] for i in crf.decode(emissions, mask=mask)[0]]
    assert tags == ["O", "B-PER"], (
        f"Orientation sai: kỳ vọng ['O','B-PER'] khi transitions[O,B-PER] "
        f"cực lớn, nhưng decode ra {tags} -- build_bioes_transition_masks() "
        f"phải dùng illegal_transition[i,j] nghĩa là 'từ i đến j'."
    )


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
