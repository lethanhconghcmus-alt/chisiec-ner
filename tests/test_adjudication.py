import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.adjudication import (
    NEVER_APPLY_DECISIONS,
    SEMANTIC_ROLE_TO_LABEL,
    build_entity_index,
    check_no_overlap_after_edit,
    process_deepdive_occurrences,
    process_guideline_ambiguities_surface_level,
    process_singleton_anomalies,
    spans_overlap,
    validate_dataset_checksums,
)


def _entity(split, sample_id, start, end, label, surface, text=None):
    return {"split": split, "sample_id": sample_id, "start": start, "end": end,
            "label": label, "surface": surface, "text": text or surface,
            "document_id": None, "entity_id": f"{split}-{sample_id}-{start}"}


# ── spans_overlap / check_no_overlap_after_edit ─────────────────────────────
def test_spans_overlap_basic():
    assert spans_overlap(0, 2, 2, 4) is True
    assert spans_overlap(0, 2, 3, 4) is False
    assert spans_overlap(5, 8, 0, 4) is False


def test_check_no_overlap_after_edit_ignores_self_and_detects_others():
    ents = [_entity("train", 0, 0, 1, "TITLE", "太師"), _entity("train", 0, 2, 4, "PER", "陳守度")]
    # sua chinh entity (2,4) -> gia span khong doi -> khong dam len chinh no
    assert check_no_overlap_after_edit(ents, 2, 4, 2, 4) is True
    # neu span moi de len entity khac (0,1) -> phat hien
    assert check_no_overlap_after_edit(ents, 1, 3, 2, 4) is False


# ── process_singleton_anomalies: LUÔN skip (thiếu structured field) ─────────
def test_process_singleton_anomalies_always_skips_with_clear_reason():
    df = pd.DataFrame([
        {"surface": "光紹帝", "reviewer_decision": "CORRECT_LABEL", "record_ref": "train:568"},
        {"surface": "王", "reviewer_decision": "KEEP_GOLD", "record_ref": "test:17"},
        {"surface": "神武", "reviewer_decision": "", "record_ref": "test:69"},
    ])
    proposed, skipped = process_singleton_anomalies(df)
    assert proposed == []
    assert len(skipped) == 3
    reasons = {s.reason for s in skipped}
    assert "missing_structured_final_label_or_offset_fields" in reasons
    assert "no_change_needed_keep_gold" in reasons
    assert "reviewer_decision_empty" in reasons


# ── process_guideline_ambiguities_surface_level: LUÔN skip ──────────────────
def test_process_guideline_ambiguities_surface_level_always_skips():
    df = pd.DataFrame([
        {"surface": "哀牢", "reviewer_decision": "GUIDELINE_AMBIGUITY"},
        {"surface": "承司", "reviewer_decision": "CORRECT_LABEL"},  # van bi skip du la CORRECT_LABEL
    ])
    skipped = process_guideline_ambiguities_surface_level(df)
    assert len(skipped) == 2
    assert all(s.reason == "surface_level_only_not_occurrence_reviewed_majority_vote_forbidden"
               for s in skipped)


# ── process_deepdive_occurrences: CORRECT_LABEL qua semantic_role mapping ──
def test_deepdive_correct_label_applies_via_semantic_role_mapping():
    entities = [_entity("train", 184, 10, 11, "ORG", "日燏", text="封佐聖太師日燏為大王")]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "entity_id": 1, "split": "train", "sample_id": 184, "document_id": None,
        "gold_surface": "日燏", "gold_start": 10, "gold_end": 11, "gold_label": "ORG",
        "full_sentence": "封佐聖太師日燏為大王",
        "semantic_role_candidate": "PERSON", "reviewer_decision": "CORRECT_LABEL",
    }])
    proposed, skipped = process_deepdive_occurrences(df, idx)
    assert skipped == []
    assert len(proposed) == 1
    c = proposed[0]
    assert c.new_label == "PER"
    assert c.old_label == "ORG"
    assert c.old_span == (10, 11) == c.new_span
    assert c.validation_status == "valid"


def test_deepdive_never_apply_decisions_are_skipped():
    entities = [_entity("train", 1, 0, 1, "ORG", "莫氏")]
    idx = build_entity_index(entities)
    rows = []
    for decision in NEVER_APPLY_DECISIONS:
        rows.append({
            "entity_id": 1, "split": "train", "sample_id": 1, "document_id": None,
            "gold_surface": "莫氏", "gold_start": 0, "gold_end": 1, "gold_label": "ORG",
            "full_sentence": "莫氏之亂", "semantic_role_candidate": "CLAN",
            "reviewer_decision": decision,
        })
    df = pd.DataFrame(rows)
    proposed, skipped = process_deepdive_occurrences(df, idx)
    assert proposed == []
    assert len(skipped) == len(NEVER_APPLY_DECISIONS)


def test_deepdive_correct_boundary_and_split_entity_skipped_missing_spec():
    entities = [_entity("train", 1, 0, 1, "TITLE", "御史")]
    idx = build_entity_index(entities)
    df = pd.DataFrame([
        {"entity_id": 1, "split": "train", "sample_id": 1, "document_id": None,
         "gold_surface": "御史", "gold_start": 0, "gold_end": 1, "gold_label": "TITLE",
         "full_sentence": "御史臺奏事", "semantic_role_candidate": "OFFICE",
         "reviewer_decision": "CORRECT_BOUNDARY"},
        {"entity_id": 2, "split": "train", "sample_id": 1, "document_id": None,
         "gold_surface": "明成化", "gold_start": 0, "gold_end": 2, "gold_label": "ORG",
         "full_sentence": "明成化十八年", "semantic_role_candidate": "OTHER",
         "reviewer_decision": "SPLIT_ENTITY"},
    ])
    proposed, skipped = process_deepdive_occurrences(df, idx)
    assert proposed == []
    assert len(skipped) == 2
    assert all("missing_structured_offset_or_split_spec_for" in s.reason for s in skipped)


def test_deepdive_semantic_role_other_not_mappable_skipped():
    entities = [_entity("train", 1, 0, 1, "ORG", "吳")]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "entity_id": 1, "split": "train", "sample_id": 1, "document_id": None,
        "gold_surface": "吳", "gold_start": 0, "gold_end": 1, "gold_label": "ORG",
        "full_sentence": "吳王", "semantic_role_candidate": "OTHER",
        "reviewer_decision": "CORRECT_LABEL",
    }])
    proposed, skipped = process_deepdive_occurrences(df, idx)
    assert proposed == []
    assert "semantic_role_candidate_not_mappable" in skipped[0].reason


def test_deepdive_stale_dataset_causes_validation_error_not_silent_apply():
    # Entity that KHONG con ton tai dung offset/label trong data hien tai
    # (gia lap dataset da doi so voi luc export workbook)
    entities = [_entity("train", 1, 0, 1, "PER", "日燏")]  # label da la PER roi, khac voi workbook ghi ORG
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "entity_id": 1, "split": "train", "sample_id": 1, "document_id": None,
        "gold_surface": "日燏", "gold_start": 0, "gold_end": 1, "gold_label": "ORG",
        "full_sentence": "日燏之功", "semantic_role_candidate": "PERSON",
        "reviewer_decision": "CORRECT_LABEL",
    }])
    proposed, skipped = process_deepdive_occurrences(df, idx)
    assert len(proposed) == 1
    assert proposed[0].validation_status == "error"
    assert "mismatch" in proposed[0].validation_message


def test_deepdive_derived_label_equal_to_gold_is_noop():
    entities = [_entity("train", 1, 0, 1, "PER", "日燏")]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "entity_id": 1, "split": "train", "sample_id": 1, "document_id": None,
        "gold_surface": "日燏", "gold_start": 0, "gold_end": 1, "gold_label": "PER",
        "full_sentence": "日燏之功", "semantic_role_candidate": "PERSON",
        "reviewer_decision": "CORRECT_LABEL",
    }])
    proposed, skipped = process_deepdive_occurrences(df, idx)
    assert proposed == []
    assert skipped[0].reason == "derived_label_equals_gold_no_change"


# ── validate_dataset_checksums ───────────────────────────────────────────────
def test_validate_dataset_checksums_detects_mismatch():
    prior = {"dataset_checksums": {"train": "aaa", "dev": "bbb"}}
    current = {"train": "aaa", "dev": "CHANGED"}
    errors = validate_dataset_checksums(prior, current)
    assert len(errors) == 1
    assert "dev" in errors[0]


def test_validate_dataset_checksums_no_error_when_matching_or_missing_prior():
    assert validate_dataset_checksums({"dataset_checksums": {"train": "x"}}, {"train": "x"}) == []
    assert validate_dataset_checksums(None, {"train": "x"}) == []
    assert validate_dataset_checksums({"dataset_checksums": {}}, {"train": "x"}) == []
