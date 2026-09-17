import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.bioes_utils import (
    BIOFormatError,
    bioes_to_spans,
    build_bioes_label_map,
    compute_boundary_pos_weight,
    convert_bio_to_bioes,
    convert_dataset_bio_to_bioes,
    derive_boundary_labels,
    find_bio_violations,
    repair_bioes_sequence,
    spans_to_bioes,
)


# ── 1. BIO -> BIOES ─────────────────────────────────────────────────────────
def test_single_token_entity_becomes_S():
    bioes, n_rep = convert_bio_to_bioes(["B-PER"])
    assert bioes == ["S-PER"]
    assert n_rep == 0


def test_two_token_entity_becomes_B_E():
    bioes, n_rep = convert_bio_to_bioes(["B-PER", "I-PER"])
    assert bioes == ["B-PER", "E-PER"]
    assert n_rep == 0


def test_three_token_entity_becomes_B_I_E():
    bioes, n_rep = convert_bio_to_bioes(["B-PER", "I-PER", "I-PER"])
    assert bioes == ["B-PER", "I-PER", "E-PER"]
    assert n_rep == 0


def test_readme_example_太師陳守度至化州():
    bio = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    expected = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "B-LOC", "E-LOC"]
    bioes, n_rep = convert_bio_to_bioes(bio)
    assert bioes == expected
    assert n_rep == 0


def test_two_adjacent_entities():
    bio = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER"]
    expected = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER"]
    bioes, _ = convert_bio_to_bioes(bio)
    assert bioes == expected


def test_all_O_sequence_unchanged():
    bio = ["O", "O", "O"]
    bioes, n_rep = convert_bio_to_bioes(bio)
    assert bioes == ["O", "O", "O"]
    assert n_rep == 0


def test_strict_mode_raises_on_orphan_I():
    bio = ["O", "I-PER", "O"]
    with pytest.raises(BIOFormatError) as exc:
        convert_bio_to_bioes(bio, sample_id=42, mode="strict")
    msg = str(exc.value)
    assert "sample_id=42" in msg
    assert "pos=1" in msg


def test_repair_mode_fixes_orphan_I_and_counts():
    bio = ["O", "I-PER", "I-PER", "O"]
    bioes, n_rep = convert_bio_to_bioes(bio, mode="repair")
    assert bioes == ["O", "B-PER", "E-PER", "O"]
    assert n_rep == 1


def test_orphan_I_wrong_type_after_B_is_repaired_as_new_entity():
    # B-PER I-LOC -> I-LOC mồ côi vì type khác -> repair thành B-LOC
    bio = ["B-PER", "I-LOC"]
    bioes, n_rep = convert_bio_to_bioes(bio, mode="repair")
    assert bioes == ["S-PER", "S-LOC"]
    assert n_rep == 1


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        convert_bio_to_bioes(["O"], mode="bogus")


def test_find_bio_violations_reports_position_and_reason():
    violations = find_bio_violations(["O", "I-LOC", "B-PER", "I-ORG"], sample_id="s1")
    assert len(violations) == 2
    assert violations[0]["position"] == 1
    assert violations[1]["position"] == 3
    assert "s1" in violations[0]["reason"]


def test_convert_dataset_bio_to_bioes_strict_and_stats():
    data = [
        (["a", "b"], ["B-PER", "I-PER"]),
        (["c"], ["O"]),
    ]
    out, stats = convert_dataset_bio_to_bioes(data, mode="strict")
    assert out[0] == (["a", "b"], ["B-PER", "E-PER"])
    assert stats["num_sequences"] == 2
    assert stats["num_sequences_repaired"] == 0


def test_convert_dataset_bio_to_bioes_repair_logs_counts():
    data = [(["a", "b"], ["O", "I-PER"])]
    out, stats = convert_dataset_bio_to_bioes(data, mode="repair")
    assert out[0][1] == ["O", "S-PER"]
    assert stats["num_sequences_repaired"] == 1
    assert stats["num_tags_repaired"] == 1


# ── 2. Boundary label derivation ────────────────────────────────────────────
def test_derive_boundary_labels_matches_spec_example():
    bioes = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "B-LOC", "E-LOC"]
    start, end = derive_boundary_labels(bioes)
    assert start == [1, 0, 1, 0, 0, 0, 1, 0]
    assert end == [0, 1, 0, 0, 1, 0, 0, 1]


def test_derive_boundary_labels_single_token_entity_start_end_both_1():
    start, end = derive_boundary_labels(["S-PER"])
    assert start == [1]
    assert end == [1]


def test_derive_boundary_labels_all_O():
    start, end = derive_boundary_labels(["O", "O"])
    assert start == [0, 0]
    assert end == [0, 0]


# ── 3. Boundary pos_weight (train-only) ─────────────────────────────────────
def test_compute_boundary_pos_weight_clips_and_counts():
    # 1 positive start, 9 negative -> pos_weight would be 9, under clip of 10
    data = [(["a"] * 10, ["S-PER"] + ["O"] * 9)]
    stats = compute_boundary_pos_weight(data, max_pos_weight=10.0)
    assert stats["start_num_positive"] == 1
    assert stats["start_num_negative"] == 9
    assert stats["start_pos_weight"] == pytest.approx(9.0)


def test_compute_boundary_pos_weight_clips_to_max():
    data = [(["a"] * 100, ["S-PER"] + ["O"] * 99)]
    stats = compute_boundary_pos_weight(data, max_pos_weight=5.0)
    assert stats["start_pos_weight"] == 5.0


def test_compute_boundary_pos_weight_zero_positive_returns_max():
    data = [(["a", "b"], ["O", "O"])]
    stats = compute_boundary_pos_weight(data, max_pos_weight=7.0)
    assert stats["start_pos_weight"] == 7.0
    assert stats["start_num_positive"] == 0


# ── 4. BIOES -> spans + round-trip ──────────────────────────────────────────
def test_bioes_to_spans_basic():
    bioes = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "S-LOC"]
    spans = bioes_to_spans(bioes)
    assert spans == [(0, 1, "TITLE"), (2, 4, "PER"), (6, 6, "LOC")]


def test_spans_roundtrip():
    bioes = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "S-LOC"]
    spans = bioes_to_spans(bioes)
    assert spans_to_bioes(spans, len(bioes)) == bioes


def test_bioes_to_spans_raises_on_dangling_I_without_repair():
    with pytest.raises(BIOFormatError):
        bioes_to_spans(["O", "I-PER", "O"])


def test_bioes_to_spans_raises_on_unclosed_entity():
    with pytest.raises(BIOFormatError):
        bioes_to_spans(["B-PER", "I-PER"])


# ── 9. CRF decode invalid-sequence repair ───────────────────────────────────
def test_repair_bioes_sequence_closes_unclosed_entity_at_end():
    repaired, n_rep = repair_bioes_sequence(["B-PER", "I-PER"])
    assert repaired == ["B-PER", "E-PER"]
    assert n_rep == 1
    assert bioes_to_spans(repaired) == [(0, 1, "PER")]


def test_repair_bioes_sequence_handles_orphan_I_mid_sequence():
    # I-PER xuất hiện không có entity mở -> coi là B-PER mới
    repaired, n_rep = repair_bioes_sequence(["O", "I-PER", "O"])
    assert repaired == ["O", "S-PER", "O"]
    # 2 correction actions: (1) coi I-PER mồ côi là B-PER mới, (2) đóng nó
    # lại thành S-PER khi gặp O ngay sau (không có E-PER).
    assert n_rep == 2


def test_repair_bioes_sequence_closes_open_entity_when_O_appears():
    # B-PER O (không có E-PER) -> đóng tại vị trí trước đó thành E-PER... nhưng
    # vị trí trước đó chính là B-PER (chỉ 1 token) -> thành S-PER
    repaired, n_rep = repair_bioes_sequence(["B-PER", "O"])
    assert repaired == ["S-PER", "O"]
    assert n_rep == 1


def test_repair_bioes_sequence_closes_open_entity_on_new_B():
    repaired, n_rep = repair_bioes_sequence(["B-PER", "I-PER", "B-LOC", "E-LOC"])
    assert repaired == ["B-PER", "E-PER", "B-LOC", "E-LOC"]
    assert n_rep == 1
    assert bioes_to_spans(repaired) == [(0, 1, "PER"), (2, 3, "LOC")]


def test_repair_bioes_sequence_noop_on_already_valid_sequence():
    valid = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "S-LOC"]
    repaired, n_rep = repair_bioes_sequence(valid)
    assert repaired == valid
    assert n_rep == 0


def test_repair_bioes_sequence_mismatched_type_continuation():
    # B-PER I-LOC -> I-LOC không khớp loại đang mở (PER) -> đóng PER tại vị
    # trí trước (thành S-PER vì chỉ 1 token), mở LOC mới tại vị trí đó
    repaired, n_rep = repair_bioes_sequence(["B-PER", "I-LOC"])
    assert repaired == ["S-PER", "S-LOC"]
    # 3 correction actions: đóng PER sớm (1) + coi I-LOC là B-LOC (1) +
    # đóng LOC ở cuối chuỗi vì không có E-LOC (1).
    assert n_rep == 3


# ── build_bioes_label_map ────────────────────────────────────────────────────
def test_build_bioes_label_map_has_21_labels_and_O_is_zero():
    label2id, id2label = build_bioes_label_map()
    assert len(label2id) == 21
    assert label2id["O"] == 0
    for etype in ["PER", "LOC", "ORG", "TITLE", "DTM"]:
        for prefix in ("B", "I", "E", "S"):
            assert f"{prefix}-{etype}" in label2id
    assert id2label[label2id["S-DTM"]] == "S-DTM"
