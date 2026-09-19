import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.normalization import (
    VERIFIED_PUA_MAPPING,
    codepoint_category,
    normalize_text_with_alignment,
    remap_span_inclusive,
)

PUA1 = chr(0xF0C51)
PUA2 = chr(0xF025F)


def test_verified_pua_mapping_normalizes_to_tai():
    text = f"御史{PUA1}都察院"
    normalized, _, _, applied = normalize_text_with_alignment(text)
    assert normalized == "御史臺都察院"
    assert len(applied) == 1
    assert applied[0]["codepoint"] == "U+F0C51"
    assert applied[0]["replacement"] == "臺"

    text2 = f"御史{PUA2}"
    normalized2, _, _, applied2 = normalize_text_with_alignment(text2)
    assert normalized2 == "御史臺"
    assert applied2[0]["codepoint"] == "U+F025F"


def test_codepoint_category_detects_pua_and_variation_selector_and_replacement_char():
    assert codepoint_category(0xF0C51) == "pua"
    assert codepoint_category(0xE000) == "pua"
    assert codepoint_category(0xFE0F) == "variation_selector"
    assert codepoint_category(0xFFFD) == "replacement_char"
    assert codepoint_category(ord("臺")) is None


def test_no_silent_character_deletion_length_never_decreases_to_zero():
    text = f"{PUA1}{PUA1}"
    normalized, _, _, _ = normalize_text_with_alignment(text)
    assert len(normalized) == 2  # moi PUA -> dung 1 ky tu thay the, khong bien mat


def test_raw_to_normalized_offset_alignment_matches_surface():
    # "御史臺" ~ raw = "御史" + PUA1, gold span goc (raw) la (0,2) inclusive
    # (toan bo 3 "ky tu" gom PUA), sau normalize van phai tro dung "御史臺"
    raw_text = f"御史{PUA1}"
    normalized_text, raw_to_norm, norm_to_raw, applied = normalize_text_with_alignment(raw_text)
    assert normalized_text == "御史臺"

    raw_start, raw_end = 0, 2  # "御","史",PUA1 -- inclusive
    n_start, n_end = remap_span_inclusive(raw_to_norm, raw_start, raw_end, len(raw_text), len(normalized_text))
    assert normalized_text[n_start:n_end + 1] == "御史臺"


def test_raw_to_normalized_alignment_for_span_entirely_before_pua_char():
    raw_text = f"甲乙{PUA1}丙"
    normalized_text, raw_to_norm, _, _ = normalize_text_with_alignment(raw_text)
    assert normalized_text == "甲乙臺丙"
    # span cho "甲乙" (raw 0..1) khong bi anh huong boi PUA phia sau
    n_start, n_end = remap_span_inclusive(raw_to_norm, 0, 1, len(raw_text), len(normalized_text))
    assert normalized_text[n_start:n_end + 1] == "甲乙"
    # span cho "丙" (raw index 3, sau PUA) van dung vi 1-1 length-preserving
    n_start2, n_end2 = remap_span_inclusive(raw_to_norm, 3, 3, len(raw_text), len(normalized_text))
    assert normalized_text[n_start2:n_end2 + 1] == "丙"


def test_normalization_is_deterministic():
    text = f"御史{PUA1}都御史{PUA2}"
    r1 = normalize_text_with_alignment(text)
    r2 = normalize_text_with_alignment(text)
    assert r1 == r2


def test_unmapped_pua_codepoint_is_left_untouched_not_guessed():
    unmapped_pua = chr(0xF0001)
    text = f"甲{unmapped_pua}乙"
    normalized, _, _, applied = normalize_text_with_alignment(text)
    assert normalized == text  # khong doan mapping cho codepoint chua verify
    assert applied == []


def test_custom_mapping_override_still_length_preserving_for_these_two_codepoints():
    normalized, raw_to_norm, norm_to_raw, applied = normalize_text_with_alignment(
        f"甲{chr(0xF0C51)}乙", mapping=VERIFIED_PUA_MAPPING,
    )
    assert normalized == "甲臺乙"
    assert len(normalized) == 3
