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


# ── process_boundary_split_fillin ──────────────────────────────────────────
def _entity_full(split, sample_id, start, end, label, surface, text):
    return {"split": split, "sample_id": sample_id, "start": start, "end": end,
            "label": label, "surface": surface, "text": text,
            "document_id": None, "entity_id": f"{split}-{sample_id}-{start}"}


def test_boundary_fillin_widens_span_correctly():
    from src.adjudication import process_boundary_split_fillin
    text = "希葛為延河伯祐豐為提刑監察御史臺奏"
    # "御史" tai vi tri 13-14 (2 ky tu), mo rong them '臺' o vi tri 15
    entities = [_entity_full("train", 17, 13, 14, "TITLE", "御史", text)]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "row_type": "CORRECT_BOUNDARY", "entity_id": "x", "split": "train", "sample_id": 17,
        "document_id": None, "gold_surface": "御史", "gold_start": 13, "gold_end": 14,
        "gold_label": "TITLE", "full_sentence": text, "final_label": "ORG",
        "final_start": 13, "final_end": 15, "split2_start": None, "fill_notes": "",
        "existing_reviewer_notes": "",
    }])
    proposed, skipped = process_boundary_split_fillin(df, idx)
    assert skipped == []
    assert len(proposed) == 1
    c = proposed[0]
    assert c.new_span == (13, 15)
    assert c.new_label == "ORG"
    assert c.validation_status == "valid"
    assert text[c.new_span[0]:c.new_span[1] + 1] == "御史臺"


def test_boundary_fillin_missing_final_fields_is_skipped():
    from src.adjudication import process_boundary_split_fillin
    entities = [_entity_full("train", 1, 0, 1, "TITLE", "御史", "御史臺奏事")]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "row_type": "CORRECT_BOUNDARY", "entity_id": "x", "split": "train", "sample_id": 1,
        "document_id": None, "gold_surface": "御史", "gold_start": 0, "gold_end": 1,
        "gold_label": "TITLE", "full_sentence": "御史臺奏事", "final_label": None,
        "final_start": None, "final_end": None, "split2_start": None,
        "fill_notes": "", "existing_reviewer_notes": "",
    }])
    proposed, skipped = process_boundary_split_fillin(df, idx)
    assert proposed == []
    assert skipped[0].reason == "final_label_or_offset_still_missing"


def test_boundary_fillin_split_entity_without_split2_treated_as_single_widen():
    from src.adjudication import process_boundary_split_fillin
    text = "明成化十八年"
    entities = [_entity_full("train", 79, 0, 2, "ORG", "明成化", text)]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "row_type": "SPLIT_ENTITY", "entity_id": "x", "split": "train", "sample_id": 79,
        "document_id": None, "gold_surface": "明成化", "gold_start": 0, "gold_end": 2,
        "gold_label": "ORG", "full_sentence": text, "final_label": "DTM",
        "final_start": 0, "final_end": 5, "split2_start": None,
        "fill_notes": "", "existing_reviewer_notes": "",
    }])
    proposed, skipped = process_boundary_split_fillin(df, idx)
    assert len(proposed) == 1  # KHONG tach thanh 2, vi split2 de trong
    assert proposed[0].new_span == (0, 5)
    assert proposed[0].new_label == "DTM"


def test_boundary_fillin_real_split_produces_two_changes():
    from src.adjudication import process_boundary_split_fillin
    text = "明成化十八年"
    entities = [_entity_full("train", 79, 0, 5, "ORG", "明成化十八年", text)]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "row_type": "SPLIT_ENTITY", "entity_id": "x", "split": "train", "sample_id": 79,
        "document_id": None, "gold_surface": "明成化十八年", "gold_start": 0, "gold_end": 5,
        "gold_label": "ORG", "full_sentence": text,
        "final_label": "ORG", "final_start": 0, "final_end": 0,
        "split2_start": 1, "split2_end": 5, "split2_label": "DTM",
        "fill_notes": "", "existing_reviewer_notes": "",
    }])
    proposed, skipped = process_boundary_split_fillin(df, idx)
    assert len(proposed) == 2
    assert proposed[0].new_span == (0, 0) and proposed[0].new_label == "ORG"
    assert proposed[1].new_span == (1, 5) and proposed[1].new_label == "DTM"
    assert all(c.validation_status == "valid" for c in proposed)


def test_boundary_fillin_stale_dataset_detected_as_error():
    from src.adjudication import process_boundary_split_fillin
    # Entity hien tai KHONG con dung o vi tri/nhan da ghi trong workbook
    entities = [_entity_full("train", 1, 0, 1, "ORG", "御史", "御史臺")]
    idx = build_entity_index(entities)
    df = pd.DataFrame([{
        "row_type": "CORRECT_BOUNDARY", "entity_id": "x", "split": "train", "sample_id": 1,
        "document_id": None, "gold_surface": "御史", "gold_start": 0, "gold_end": 1,
        "gold_label": "TITLE",  # workbook ghi TITLE nhung live data la ORG
        "full_sentence": "御史臺", "final_label": "ORG",
        "final_start": 0, "final_end": 2, "split2_start": None,
        "fill_notes": "", "existing_reviewer_notes": "",
    }])
    proposed, skipped = process_boundary_split_fillin(df, idx)
    assert len(proposed) == 1
    assert proposed[0].validation_status == "error"
    assert "mismatch" in proposed[0].validation_message


# ── MERGE_ENTITY generic support ─────────────────────────────────────────────
from src.adjudication import build_entity_by_id, process_collision_candidates, validate_and_build_merge


def _merge_row(collision_id, split, sample_id, merged_ids, resulting_start, resulting_end,
                resulting_label, expected_surface="", rule_id="GR-TEST", notes="test note",
                action="MERGE_ENTITY"):
    return {
        "collision_id": collision_id, "split": split, "sample_id": sample_id,
        "document_id": None, "selected_action": action,
        "merged_entity_ids": ";".join(merged_ids),
        "resulting_start": resulting_start, "resulting_end": resulting_end,
        "resulting_label": resulting_label, "expected_resulting_surface": expected_surface,
        "guideline_rule_id": rule_id, "reviewer_notes": notes,
    }


def test_merge_two_adjacent_dtm_valid():
    text = "明成化十七年春"
    entities = [
        _entity_full("train", 1, 0, 2, "ORG", "明成化", text),
        _entity_full("train", 1, 3, 5, "DTM", "十七年", text),
    ]
    entities[0]["entity_id"] = "e1"
    entities[1]["entity_id"] = "e2"
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(1, "train", 1, ["e1", "e2"], 0, 5, "DTM", "明成化十七年")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert skipped == []
    assert len(proposed) == 1
    m = proposed[0]
    assert m.validation_status == "valid"
    assert m.resulting_span == (0, 5)
    assert m.resulting_label == "DTM"
    assert set(m.source_entity_ids) == {"e1", "e2"}


def test_merge_two_adjacent_per_valid():
    text = "陳太宗即位"
    entities = [
        _entity_full("train", 2, 0, 0, "ORG", "陳", text),
        _entity_full("train", 2, 1, 2, "PER", "太宗", text),
    ]
    entities[0]["entity_id"] = "a1"
    entities[1]["entity_id"] = "a2"
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(2, "train", 2, ["a1", "a2"], 0, 2, "PER", "陳太宗")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert len(proposed) == 1
    assert proposed[0].validation_status == "valid"


def test_merge_invalid_when_would_overlap_third_entity():
    text = "故都御史臺奏"
    entities = [
        _entity_full("train", 3, 0, 1, "LOC", "故都", text),
        _entity_full("train", 3, 2, 3, "TITLE", "御史", text),
        _entity_full("train", 3, 4, 4, "ORG", "臺", text),
    ]
    entities[0]["entity_id"] = "b1"
    entities[1]["entity_id"] = "b2"
    entities[2]["entity_id"] = "b3"
    idx = build_entity_index(entities)
    # merge b1+b2 (0..3) se de len b3 (4,4)? khong, thu truong hop de len that:
    # merge chi b2 (2,3) mo rong sang (1,3) se de len b1 (0,1) vi 1 nam trong (0,1)
    df = pd.DataFrame([_merge_row(3, "train", 3, ["b2"], 1, 3, "TITLE", "都御史")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    # merged_entity_ids chi co 1 -> khong du 2 nguon
    assert proposed == []
    assert skipped[0].reason == "merged_entity_ids_missing_or_less_than_2"


def test_merge_invalid_source_entities_different_sample():
    entities = [
        _entity_full("train", 1, 0, 1, "DTM", "明成", "明成化"),
        _entity_full("train", 2, 0, 1, "DTM", "化十", "化十年"),
    ]
    entities[0]["entity_id"] = "c1"
    entities[1]["entity_id"] = "c2"
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(4, "train", 1, ["c1", "c2"], 0, 1, "DTM", "??")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert len(proposed) == 1
    assert proposed[0].validation_status == "error"
    assert "not_all_same_sample" in proposed[0].validation_message


def test_merge_not_applied_when_selected_action_blank():
    entities = [_entity_full("train", 1, 0, 1, "DTM", "明成", "明成化十七年")]
    entities[0]["entity_id"] = "d1"
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(5, "train", 1, ["d1"], 0, 1, "DTM", action="")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert proposed == []
    assert skipped[0].reason == "pending_review"


def test_merge_missing_guideline_rule_id_or_notes_rejected():
    text = "明成化十七年"
    entities = [
        _entity_full("train", 1, 0, 2, "ORG", "明成化", text),
        _entity_full("train", 1, 3, 5, "DTM", "十七年", text),
    ]
    entities[0]["entity_id"] = "e1"
    entities[1]["entity_id"] = "e2"
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(6, "train", 1, ["e1", "e2"], 0, 5, "DTM", rule_id="", notes="")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert proposed == []
    assert skipped[0].reason in ("missing_guideline_rule_id",)  # kiem tra dau tien trong ham


def test_merge_atomic_sources_disappear_result_is_single_entity():
    """Test 'atomic': logic downstream (khi that su apply -- chua implement
    o vong nay) phai the hien merged entity thay THE cho ca 2 nguon, khong
    con giu nguyen 2 nguon cu. O muc validate, ta kiem tra source_entity_ids
    duoc ghi day du va resulting_span duy nhat bao trum ca 2."""
    text = "明成化十七年"
    entities = [
        _entity_full("train", 1, 0, 2, "ORG", "明成化", text),
        _entity_full("train", 1, 3, 5, "DTM", "十七年", text),
    ]
    entities[0]["entity_id"] = "e1"
    entities[1]["entity_id"] = "e2"
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(7, "train", 1, ["e1", "e2"], 0, 5, "DTM", "明成化十七年")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    m = proposed[0]
    assert m.validation_status == "valid"
    assert len(m.source_spans) == 2
    assert m.resulting_span == (0, 5)
    # Khong co "third" span nao con lai trong ket qua -- chi 1 resulting span duy nhat


def test_dataset_v1_immutable_process_functions_never_mutate_entities():
    text = "明成化十七年"
    entities = [
        _entity_full("train", 1, 0, 2, "ORG", "明成化", text),
        _entity_full("train", 1, 3, 5, "DTM", "十七年", text),
    ]
    entities[0]["entity_id"] = "e1"
    entities[1]["entity_id"] = "e2"
    import copy
    snapshot = copy.deepcopy(entities)
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(8, "train", 1, ["e1", "e2"], 0, 5, "DTM", "明成化十七年")])
    process_collision_candidates(df, entities, idx)
    assert entities == snapshot


def test_merge_allows_small_gap_between_sources_for_split_name_reconstruction():
    # 黎(PER,33) ... 輔(khong gan nhan, 34) ... 陳(PER,35) -> gop thanh 黎輔陳/PER
    text_prefix = "x" * 33
    text = text_prefix + "黎輔陳" + "y" * 5
    entities = [
        _entity_full("train", 107, 33, 33, "PER", "黎", text),
        _entity_full("train", 107, 35, 35, "PER", "陳", text),
    ]
    entities[0]["entity_id"] = 1567
    entities[1]["entity_id"] = 1568
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(9, "train", 107, ["1567", "1568"], 33, 35, "PER",
                                    "黎輔陳", rule_id="GR-12", notes="ten bi cat lam doi")])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert len(proposed) == 1
    assert proposed[0].validation_status == "valid"
    assert proposed[0].resulting_span == (33, 35)


def test_merge_rejects_gap_too_large_between_sources():
    text = "黎" + "z" * 20 + "陳"
    entities = [
        _entity_full("train", 1, 0, 0, "PER", "黎", text),
        _entity_full("train", 1, 21, 21, "PER", "陳", text),
    ]
    entities[0]["entity_id"] = 9001
    entities[1]["entity_id"] = 9002
    idx = build_entity_index(entities)
    df = pd.DataFrame([_merge_row(10, "train", 1, ["9001", "9002"], 0, 21, "PER", text)])
    proposed, skipped = process_collision_candidates(df, entities, idx)
    assert len(proposed) == 1
    assert proposed[0].validation_status == "error"
    assert "gap_too_large" in proposed[0].validation_message
