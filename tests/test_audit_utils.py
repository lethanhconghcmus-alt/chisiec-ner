import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit_utils import (
    REVIEW_COLUMNS,
    build_admin_suffix_boundary_report,
    build_ambiguity_report,
    build_confusable_pair_report,
    build_merge_split_report,
    build_review_workbook,
    bio_to_entities,
    compute_priority_score,
    extract_all_spans,
    normalize_surface,
)


# ── 1. BIO -> entities ──────────────────────────────────────────────────────
def test_bio_to_entities_basic_multi_token():
    tokens = list("陳守度")
    labels = ["B-PER", "I-PER", "I-PER"]
    ents = bio_to_entities(tokens, labels)
    assert ents == [{"start": 0, "end": 2, "type": "PER", "violation": False}]


def test_bio_to_entities_one_character_entity():
    tokens = list("王在位")
    labels = ["B-PER", "O", "O"]
    ents = bio_to_entities(tokens, labels)
    assert ents == [{"start": 0, "end": 0, "type": "PER", "violation": False}]


def test_bio_to_entities_adjacent_entities_no_O_between():
    tokens = list("太師陳守度")
    labels = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER"]
    ents = bio_to_entities(tokens, labels)
    assert ents == [
        {"start": 0, "end": 1, "type": "TITLE", "violation": False},
        {"start": 2, "end": 4, "type": "PER", "violation": False},
    ]


def test_bio_to_entities_invalid_orphan_I_flagged_not_dropped():
    tokens = list("蠻人遁去")
    labels = ["O", "I-PER", "O", "O"]  # I-PER mo coi, khong co B-PER truoc
    ents = bio_to_entities(tokens, labels)
    assert len(ents) == 1
    assert ents[0]["type"] == "PER"
    assert ents[0]["start"] == 1 and ents[0]["end"] == 1
    assert ents[0]["violation"] is True


def test_bio_to_entities_mismatched_type_I_splits_into_two_entities():
    # B-PER roi I-LOC (khac loai) -- PER dung o vi tri 0, LOC (violation) o vi tri 1
    tokens = list("蠻人")
    labels = ["B-PER", "I-LOC"]
    ents = bio_to_entities(tokens, labels)
    assert ents == [
        {"start": 0, "end": 0, "type": "PER", "violation": False},
        {"start": 1, "end": 1, "type": "LOC", "violation": True},
    ]


def test_bio_to_entities_supports_bioes_S_and_E():
    tokens = list("太師陳")
    labels = ["B-TITLE", "E-TITLE", "S-PER"]
    ents = bio_to_entities(tokens, labels)
    assert ents == [
        {"start": 0, "end": 1, "type": "TITLE", "violation": False},
        {"start": 2, "end": 2, "type": "PER", "violation": False},
    ]


# ── Unicode/offset consistency: text[start:end+1] == surface ───────────────
def test_extracted_span_offsets_match_surface_exactly():
    tokens = list("太師陳守度至化州")
    labels = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    splits = {"train": [(tokens, labels)]}
    entities = extract_all_spans(splits)
    text = "".join(tokens)
    for e in entities:
        assert text[e["start"]: e["end"] + 1] == e["surface"]
    assert [e["surface"] for e in entities] == ["太師", "陳守度", "化州"]


def test_extracted_span_offsets_consistent_across_multiple_sentences():
    data = [
        (list("周㐫貪諸將"), ["B-ORG", "O", "O", "O", "O"]),
        (list("蠻人遁去者"), ["B-PER", "O", "O", "O", "O"]),
    ]
    splits = {"train": data}
    entities = extract_all_spans(splits)
    for e in entities:
        text = e["text"]
        assert text[e["start"]: e["end"] + 1] == e["surface"]


# ── normalize_surface ────────────────────────────────────────────────────────
def test_normalize_surface_strips_whitespace_only():
    assert normalize_surface("  化州 ") == "化州"
    assert normalize_surface("化州") == "化州"


# ── 3. Ambiguity report ──────────────────────────────────────────────────────
def test_ambiguity_report_filters_to_multi_label_surfaces_only():
    entities = [
        {"surface": "禮部", "label": "ORG", "split": "train", "sample_id": 0,
         "start": 0, "end": 1, "context": "c1", "document_id": None, "entity_id": 0},
        {"surface": "禮部", "label": "TITLE", "split": "dev", "sample_id": 1,
         "start": 0, "end": 1, "context": "c2", "document_id": None, "entity_id": 1},
        {"surface": "太師", "label": "TITLE", "split": "train", "sample_id": 2,
         "start": 0, "end": 1, "context": "c3", "document_id": None, "entity_id": 2},
    ]
    report = build_ambiguity_report(entities)
    surfaces = {r["surface"] for r in report}
    assert surfaces == {"禮部"}  # 太師 chi co 1 nhan -> bi loc
    assert report[0]["num_unique_labels"] == 2
    assert report[0]["total_occurrences"] == 2
    assert set(report[0]["unique_labels"]) == {"ORG", "TITLE"}
    assert report[0]["splits_present"] == ["dev", "train"]


def test_ambiguity_report_sort_order_labels_then_occurrences():
    def mk(surface, label, sid):
        return {"surface": surface, "label": label, "split": "train", "sample_id": sid,
                "start": 0, "end": 0, "context": "", "document_id": None, "entity_id": sid}

    entities = [
        mk("A", "ORG", 0), mk("A", "TITLE", 1),                      # 2 labels, 2 occ
        mk("B", "ORG", 2), mk("B", "TITLE", 3), mk("B", "LOC", 4),    # 3 labels, 3 occ
        mk("C", "ORG", 5), mk("C", "TITLE", 6), mk("C", "ORG", 7),    # 2 labels, 3 occ
    ]
    report = build_ambiguity_report(entities)
    assert [r["surface"] for r in report] == ["B", "C", "A"]


# ── 4. Boundary admin-suffix report ─────────────────────────────────────────
def test_admin_suffix_boundary_report_detects_inconsistent_stem():
    entities = [
        {"surface": "化州", "label": "LOC", "split": "train", "sample_id": 0,
         "start": 0, "end": 1, "context": "c1", "entity_id": 0},
        {"surface": "化", "label": "LOC", "split": "test", "sample_id": 1,
         "start": 0, "end": 0, "context": "c2", "entity_id": 1},
        {"surface": "京畿", "label": "LOC", "split": "train", "sample_id": 2,
         "start": 0, "end": 1, "context": "c3", "entity_id": 2},  # khong co dang suffix khac -> khong bao cao
    ]
    report = build_admin_suffix_boundary_report(entities)
    stems = {r["stem"] for r in report}
    assert "化" in stems
    assert "京畿" not in stems  # chi co 1 dang, khong phai inconsistency
    rec = next(r for r in report if r["stem"] == "化")
    assert rec["num_with_suffix"] == 1 and rec["num_without_suffix"] == 1


def test_admin_suffix_boundary_report_empty_when_only_one_form_exists():
    entities = [
        {"surface": "化州", "label": "LOC", "split": "train", "sample_id": 0,
         "start": 0, "end": 1, "context": "c", "entity_id": 0},
        {"surface": "河中營", "label": "LOC", "split": "train", "sample_id": 1,
         "start": 0, "end": 2, "context": "c", "entity_id": 1},
    ]
    report = build_admin_suffix_boundary_report(entities)
    assert report == []


# ── 4b. Merge/split report (TITLE + PER adjacency style) ────────────────────
def test_merge_split_report_detects_split_and_merged_evidence():
    # "吏部" (ORG) tach voi "尚書" (TITLE) lien ke o 1 cau, va gop thanh
    # "吏部尚書" (mot TITLE dai hon) o cau khac -- dung mau hinh da biet
    # trong du an (xem memory dvsktt-sft-pilot-27b).
    org_split_sentence_tokens = list("除吏部尚書事")
    org_split_labels = ["O", "B-ORG", "I-ORG", "B-TITLE", "I-TITLE", "O"]
    merged_sentence_tokens = list("拜吏部尚書公")
    merged_labels = ["O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O"]

    splits = {
        "train": [
            (org_split_sentence_tokens, org_split_labels),
            (merged_sentence_tokens, merged_labels),
        ]
    }
    entities = extract_all_spans(splits)
    report = build_merge_split_report(entities, "ORG", "TITLE")
    surfaces = {c["surface"] for c in report["candidates"]}
    assert "吏部" in surfaces
    cand = next(c for c in report["candidates"] if c["surface"] == "吏部")
    assert cand["split_adjacent_count"] >= 1
    assert "吏部尚書" in cand["merged_forms"]


def test_merge_split_report_no_candidates_when_only_split_or_only_merged():
    tokens = list("吏部尚書")
    labels = ["B-ORG", "I-ORG", "B-TITLE", "I-TITLE"]
    splits = {"train": [(tokens, labels)]}
    entities = extract_all_spans(splits)
    report = build_merge_split_report(entities, "ORG", "TITLE")
    assert report["candidates"] == []  # chi co split, khong co merged evidence


# ── 5. Confusable pair report ───────────────────────────────────────────────
def test_confusable_pair_report_finds_org_title_overlap():
    entities = [
        {"surface": "禮部", "label": "ORG", "split": "train", "sample_id": 0,
         "start": 0, "end": 1, "context": "c1", "document_id": None, "entity_id": 0},
        {"surface": "禮部", "label": "TITLE", "split": "dev", "sample_id": 1,
         "start": 0, "end": 1, "context": "c2", "document_id": None, "entity_id": 1},
    ]
    report = build_confusable_pair_report(entities, pairs=[("ORG", "TITLE")])
    assert len(report) == 1
    assert report[0]["pair"] == ("ORG", "TITLE")
    assert len(report[0]["surfaces"]) == 1
    assert report[0]["surfaces"][0]["surface"] == "禮部"


# ── 8. Priority score heuristic ─────────────────────────────────────────────
def test_priority_score_higher_for_more_labels_and_rare_type():
    high = compute_priority_score(num_unique_labels=3, occurrence_count=10, label="ORG")
    low = compute_priority_score(num_unique_labels=0, occurrence_count=1, label="PER")
    assert high > low


def test_priority_score_boundary_and_adjacency_bonuses_additive():
    base = compute_priority_score()
    with_boundary = compute_priority_score(is_boundary_admin_issue=True)
    with_adjacency = compute_priority_score(is_title_per_adjacency=True)
    assert with_boundary > base
    assert with_adjacency > base


# ── 7. Review workbook ──────────────────────────────────────────────────────
def test_review_workbook_has_required_columns_and_no_reviewer_decision_prefilled():
    tokens = list("太師陳守度至化州")
    labels = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    tokens2 = list("化在南方")
    labels2 = ["B-LOC", "O", "O", "O"]
    splits = {"train": [(tokens, labels)], "dev": [(tokens2, labels2)]}
    entities = extract_all_spans(splits)

    ambiguity = build_ambiguity_report(entities)
    boundary = build_admin_suffix_boundary_report(entities)
    merge_split = {}

    rows = build_review_workbook(entities, ambiguity, boundary, merge_split, top_n=50)
    assert len(rows) >= 1
    for row in rows:
        assert set(row.keys()) == set(REVIEW_COLUMNS)
        assert row["reviewer_decision"] == ""
        assert row["final_label"] == ""

    boundary_rows = [r for r in rows if "boundary_admin_suffix" in r["issue_type"]]
    assert len(boundary_rows) >= 1


def test_review_workbook_sorted_by_priority_score_descending():
    tokens = list("太師陳守度至化州")
    labels = ["B-TITLE", "I-TITLE", "B-PER", "I-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    splits = {"train": [(tokens, labels)]}
    entities = extract_all_spans(splits)
    ambiguity = build_ambiguity_report(entities)
    boundary = build_admin_suffix_boundary_report(entities)
    rows = build_review_workbook(entities, ambiguity, boundary, {}, top_n=50)
    scores = [r["priority_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_review_workbook_respects_top_n_cap():
    entities = []
    ambiguity = []
    boundary = []
    for i in range(10):
        surface = "化"
        e_with = {"entity_id": i * 2, "split": "train", "sample_id": i, "document_id": None,
                   "text": "化州甲", "start": 0, "end": 1, "surface": "化州", "label": "LOC",
                   "context": "化州甲", "marked_context": "【化州】甲"}
        e_without = {"entity_id": i * 2 + 1, "split": "train", "sample_id": i, "document_id": None,
                      "text": "化甲乙", "start": 0, "end": 0, "surface": "化", "label": "LOC",
                      "context": "化甲乙", "marked_context": "【化】甲乙"}
        entities.extend([e_with, e_without])
    from src.audit_utils import build_admin_suffix_boundary_report as build_bd
    boundary = build_bd(entities)
    rows = build_review_workbook(entities, [], boundary, {}, top_n=3)
    assert len(rows) == 3
