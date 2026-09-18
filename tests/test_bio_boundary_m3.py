import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchcrf import CRF

from src.bioes_utils import (
    bio_legal_end,
    bio_legal_start,
    bio_legal_transition,
    build_transition_masks,
    derive_boundary_labels_from_bio,
)
from src.data_utils import build_label_map


def test_derive_boundary_labels_from_bio_matches_readme_example():
    bio = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    start, end = derive_boundary_labels_from_bio(bio)
    assert start == [1, 0, 1, 0, 0, 0, 1, 0]
    assert end == [0, 1, 0, 0, 1, 0, 0, 1]


def test_derive_boundary_labels_from_bio_single_token_entity():
    start, end = derive_boundary_labels_from_bio(["B-PER"])
    assert start == [1] and end == [1]


def test_derive_boundary_labels_from_bio_all_O():
    start, end = derive_boundary_labels_from_bio(["O", "O"])
    assert start == [0, 0] and end == [0, 0]


def test_derive_boundary_labels_from_bio_two_adjacent_entities_no_O_between():
    # B-PER I-PER B-LOC -- entity PER ket thuc ngay truoc B-LOC (khong O)
    start, end = derive_boundary_labels_from_bio(["B-PER", "I-PER", "B-LOC"])
    assert start == [1, 0, 1]
    assert end == [0, 1, 1]  # I-PER dong vi token ke tiep khac loai; B-LOC dong vi cuoi chuoi


# ── BIO transition rules: KHÁC BIOES ở chỗ B-X/I-X được phép "buông" ────────
def test_bio_legal_transition_allows_dangling_close_without_E_marker():
    assert bio_legal_transition("B-PER", "O") is True      # 1-token entity, hop le trong BIO
    assert bio_legal_transition("B-PER", "B-LOC") is True   # buong sang entity moi
    assert bio_legal_transition("I-PER", "O") is True
    assert bio_legal_transition("I-PER", "B-LOC") is True


def test_bio_legal_transition_still_blocks_orphan_and_cross_type_I():
    assert bio_legal_transition("O", "I-PER") is False
    assert bio_legal_transition("B-PER", "I-LOC") is False
    assert bio_legal_transition("I-PER", "I-LOC") is False
    assert bio_legal_transition("B-PER", "I-PER") is True
    assert bio_legal_transition("I-PER", "I-PER") is True


def test_bio_legal_start_end():
    assert bio_legal_start("O") and bio_legal_start("B-PER")
    assert not bio_legal_start("I-PER")
    # BIO khong co end constraint -- moi tag deu hop le o cuoi
    assert bio_legal_end("O") and bio_legal_end("B-PER") and bio_legal_end("I-PER")


def test_build_transition_masks_bio_scheme_uses_11_labels_and_allows_B_to_O():
    train_data = [(["a"], ["O"]), (["a"], ["B-PER"]), (["a"], ["I-PER"]),
                  (["a"], ["B-LOC"]), (["a"], ["I-LOC"]), (["a"], ["B-ORG"]),
                  (["a"], ["I-ORG"]), (["a"], ["B-TITLE"]), (["a"], ["I-TITLE"]),
                  (["a"], ["B-DTM"]), (["a"], ["I-DTM"])]
    label2id, id2label = build_label_map(train_data)
    assert len(label2id) == 11  # O + 2*5

    illegal_t, illegal_s, illegal_e = build_transition_masks(label2id, scheme="bio")
    assert illegal_t.shape == (11, 11)
    bper, o = label2id["B-PER"], label2id["O"]
    iper = label2id["I-PER"]
    iloc = label2id["I-LOC"]

    assert not bool(illegal_t[bper, o])     # B-PER -> O hop le trong BIO
    assert not bool(illegal_t[bper, iper])  # B-PER -> I-PER hop le
    assert bool(illegal_t[bper, iloc])      # B-PER -> I-LOC (khac loai) bat hop le
    assert bool(illegal_t[o, iper])         # O -> I-PER bat hop le (orphan)
    assert bool(illegal_s[iper])            # khong duoc bat dau bang I-*
    assert not bool(illegal_e[bper])        # BIO: B-* hop le o cuoi (khac BIOES)
    assert not bool(illegal_e[iper])


def test_build_transition_masks_bio_scheme_viterbi_never_orphans_I():
    train_data = [(["a"], [t]) for t in
                  ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]]
    label2id, id2label = build_label_map(train_data)
    from src.bioes_utils import apply_hard_transition_constraints
    n = len(label2id)
    crf = CRF(n, batch_first=True)
    illegal_t, illegal_s, illegal_e = build_transition_masks(label2id, scheme="bio")
    apply_hard_transition_constraints(crf, illegal_t, illegal_s, illegal_e)

    emissions = torch.zeros(1, 2, n)
    emissions[:, 0, label2id["I-PER"]] = 1e9  # ep cuc manh I-PER lam token dau
    mask = torch.ones(1, 2, dtype=torch.bool)
    tags = [id2label[i] for i in crf.decode(emissions, mask=mask)[0]]
    assert not tags[0].startswith("I-"), f"I-* van xuat hien o dau: {tags}"
