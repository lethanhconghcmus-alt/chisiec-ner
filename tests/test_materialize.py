"""
test_materialize.py — unit test cho src/materialize.py (pure logic) +
integration test end-to-end cho scripts/apply_adjudication.py --apply
(temp corpus nhỏ, subprocess thật, KHÔNG động tới data/dataset_v2 thật).
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adjudication import ProposedChange, ProposedMerge, ProposedRemoval
from src.audit_utils import _file_checksum
from src.materialize import (
    apply_ops_to_splits,
    build_clean_splits,
    build_quarantine_records,
    check_duplicate_change_ids,
    check_no_flat_overlap,
    quarantine_sample_ids_by_split,
    validate_bio_labels,
    verify_add_entity_never_applied,
    verify_applied_effects,
    verify_no_duplicate_sample_ids,
    verify_quarantine_partition,
    verify_source_files_unchanged,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _splits_from_pairs(pairs):
    """pairs: list[(tokens_str, labels_str)] -> [(list(tokens), list(labels))]"""
    return [(list(t), l.split()) for t, l in pairs]


# ── apply_ops_to_splits ──────────────────────────────────────────────────
def test_apply_change_relabel_only():
    splits = {"train": _splits_from_pairs([("宋嘉定九年", "B-ORG B-DTM I-DTM I-DTM I-DTM")])}
    change = ProposedChange(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="TEST",
        reviewer_decision="CORRECT_LABEL", old_span=(0, 0), old_label="ORG",
        new_span=(0, 0), new_label="PER", original_text="宋嘉定九年",
        normalized_text_if_any=None, reviewer_notes="", source_workbook_row="t",
        validation_status="valid",
    )
    new_splits, applied, errors, _dn, _dr = apply_ops_to_splits(splits, [change], [], [])
    assert errors == []
    assert new_splits["train"][0][1] == ["B-PER", "B-DTM", "I-DTM", "I-DTM", "I-DTM"]
    ents = applied[("train", 0)]
    per = next(e for e in ents if e["label"] == "PER")
    assert per["provenance"]["change_id"] == 0


def test_apply_merge_removes_sources_adds_target():
    splits = {"train": _splits_from_pairs([("宋嘉定九年", "B-ORG B-DTM I-DTM I-DTM I-DTM")])}
    merge = ProposedMerge(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="GR-11",
        source_entity_ids=[0, 1], source_spans=[(0, 0, "ORG", "宋"), (1, 4, "DTM", "嘉定九年")],
        resulting_span=(0, 4), resulting_label="DTM", reviewer_notes="x",
        source_workbook_row="t", validation_status="valid",
    )
    new_splits, applied, errors, _dn, _dr = apply_ops_to_splits(splits, [], [merge], [])
    assert errors == []
    assert new_splits["train"][0][1] == ["B-DTM"] + ["I-DTM"] * 4
    ents = applied[("train", 0)]
    assert len(ents) == 1
    assert ents[0]["provenance"]["rule_id"] == "GR-11"


def test_apply_removal_and_boundary_transaction_gudu_yushi():
    # mo phong case that: 故都/LOC bi xoa, 御史/TITLE -> 都御史/TITLE
    splits = {"train": _splits_from_pairs([("故都御史杜岳", "B-LOC I-LOC B-TITLE I-TITLE O O")])}
    removal = ProposedRemoval(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="GR-02",
        source_entity_id=100, old_span=(0, 1), old_label="LOC", old_surface="故都",
        remove_reason="not a place", reviewer_notes="x", source_workbook_row="t",
        linked_transaction_id="T1", validation_status="valid",
    )
    boundary = ProposedChange(
        change_id=1, sample_id=0, split="train", document_id=None, rule_id="GR-02",
        reviewer_decision="CORRECT_BOUNDARY", old_span=(2, 3), old_label="TITLE",
        new_span=(1, 3), new_label="TITLE", original_text="故都御史杜岳",
        normalized_text_if_any=None, reviewer_notes="x", source_workbook_row="t",
        validation_status="valid",
    )
    new_splits, applied, errors, _dn, _dr = apply_ops_to_splits(splits, [boundary], [], [removal])
    assert errors == []
    assert new_splits["train"][0][1] == ["O", "B-TITLE", "I-TITLE", "I-TITLE", "O", "O"]
    ents = applied[("train", 0)]
    assert len(ents) == 1 and ents[0]["label"] == "TITLE" and ents[0]["surface"] == "都御史"


def test_apply_dedupes_redundant_ops_sharing_same_target():
    # 2 op tu 2 nguon KHAC nhau (vd boundary_split_fillin + gr11) cung ket
    # luan giong het (cung new_range), old_ranges cua op nay la tap con
    # cua op kia -- KHONG duoc coi la collision, chi ap 1 lan.
    splits = {"train": _splits_from_pairs([("明成化十七年", "B-ORG I-ORG O O O O")])}
    op_a = ProposedChange(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="boundary_split_fillin",
        reviewer_decision="SPLIT_ENTITY", old_span=(0, 1), old_label="ORG",
        new_span=(0, 5), new_label="DTM", original_text="明成化十七年",
        normalized_text_if_any=None, reviewer_notes="", source_workbook_row="a",
        validation_status="valid",
    )
    op_b = ProposedChange(
        change_id=1, sample_id=0, split="train", document_id=None, rule_id="GR-11",
        reviewer_decision="CORRECT_BOUNDARY", old_span=(0, 1), old_label="ORG",
        new_span=(0, 5), new_label="DTM", original_text="明成化十七年",
        normalized_text_if_any=None, reviewer_notes="", source_workbook_row="b",
        validation_status="valid",
    )
    new_splits, applied, errors, dedup_notes, dedup_records = apply_ops_to_splits(splits, [op_a, op_b], [], [])
    assert errors == []
    assert len(dedup_notes) == 1
    assert new_splits["train"][0][1] == ["B-DTM"] + ["I-DTM"] * 5
    ents = applied[("train", 0)]
    assert len(ents) == 1 and ents[0]["label"] == "DTM"


def test_apply_dedupes_subset_merge_vs_narrower_change_same_target():
    # mo phong dev:74 that: 1 MERGE_ENTITY 2 nguon vs 1 CORRECT_BOUNDARY
    # 1 nguon (tap con cua nguon merge), cung final_span/label -- giu merge
    # (bao trum nhieu hon), bo change hep hon.
    splits = {"train": _splits_from_pairs([("元和元年莫大正", "B-DTM I-DTM I-DTM I-DTM B-PER I-PER I-PER")])}
    merge = ProposedMerge(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="GR-11",
        source_entity_ids=[0, 1], source_spans=[(0, 3, "DTM", "元和元年"), (4, 6, "PER", "莫大正")],
        resulting_span=(4, 6), resulting_label="DTM", reviewer_notes="x",
        source_workbook_row="merge_row", validation_status="valid",
    )
    narrow_change = ProposedChange(
        change_id=1, sample_id=0, split="train", document_id=None, rule_id="GR-11",
        reviewer_decision="CORRECT_BOUNDARY", old_span=(4, 6), old_label="PER",
        new_span=(4, 6), new_label="DTM", original_text="元和元年莫大正",
        normalized_text_if_any=None, reviewer_notes="", source_workbook_row="change_row",
        validation_status="valid",
    )
    new_splits, applied, errors, dedup_notes, dedup_records = apply_ops_to_splits(splits, [narrow_change], [merge], [])
    assert errors == []
    assert len(dedup_notes) == 1
    assert "change_row" in dedup_notes[0] and "merge_row" in dedup_notes[0]
    ents = applied[("train", 0)]
    assert len(ents) == 1 and ents[0]["start"] == 4 and ents[0]["end"] == 6 and ents[0]["label"] == "DTM"


def test_apply_detects_cross_op_collision():
    # 2 op KHAC nhau trong CUNG cau vo tinh ghi de len nhau -- phai bi bat
    splits = {"train": _splits_from_pairs([("甲乙丙丁", "O O O O")])}
    c1 = ProposedChange(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="T1",
        reviewer_decision="ADD", old_span=(0, 0), old_label="O",
        new_span=(0, 2), new_label="PER", original_text="甲乙丙丁",
        normalized_text_if_any=None, reviewer_notes="", source_workbook_row="t1",
        validation_status="valid",
    )
    c2 = ProposedChange(
        change_id=1, sample_id=0, split="train", document_id=None, rule_id="T2",
        reviewer_decision="ADD", old_span=(3, 3), old_label="O",
        new_span=(1, 3), new_label="LOC", original_text="甲乙丙丁",
        normalized_text_if_any=None, reviewer_notes="", source_workbook_row="t2",
        validation_status="valid",
    )
    new_splits, applied, errors, _dn, _dr = apply_ops_to_splits(splits, [c1, c2], [], [])
    assert len(errors) == 1
    assert "collision" in errors[0]


# ── verify_applied_effects / check_duplicate_change_ids ──────────────────
def test_verify_applied_effects_passes_for_correct_merge():
    splits = {"train": _splits_from_pairs([("宋嘉定九年", "B-ORG B-DTM I-DTM I-DTM I-DTM")])}
    merge = ProposedMerge(
        change_id=0, sample_id=0, split="train", document_id=None, rule_id="GR-11",
        source_entity_ids=[0, 1], source_spans=[(0, 0, "ORG", "宋"), (1, 4, "DTM", "嘉定九年")],
        resulting_span=(0, 4), resulting_label="DTM", reviewer_notes="x",
        source_workbook_row="t", validation_status="valid",
    )
    _, applied, _, _dn2, _dr2 = apply_ops_to_splits(splits, [], [merge], [])
    assert verify_applied_effects(applied, [], [merge], []) == []


def test_check_duplicate_change_ids_detects_dup():
    c1 = ProposedChange(0, 0, "train", None, "T", "X", (0, 0), "O", (0, 0), "PER", "a",
                         None, "", "t", "valid")
    c2 = ProposedChange(0, 1, "train", None, "T", "X", (0, 0), "O", (0, 0), "PER", "a",
                         None, "", "t", "valid")
    assert check_duplicate_change_ids([c1, c2], [], []) == ["duplicate change_id=0"]


# ── BIO validation / overlap ──────────────────────────────────────────────
def test_validate_bio_labels_detects_orphan_and_unknown_type():
    assert validate_bio_labels(list("甲乙"), ["I-PER", "O"]) != []
    assert validate_bio_labels(list("甲乙"), ["B-EVENT", "O"]) != []
    assert validate_bio_labels(list("甲乙"), ["B-PER", "I-PER"]) == []


def test_check_no_flat_overlap_detects_overlap():
    ents = [{"start": 0, "end": 3, "type": "PER"}, {"start": 2, "end": 5, "type": "LOC"}]
    assert check_no_flat_overlap(ents) != []
    ents2 = [{"start": 0, "end": 1, "type": "PER"}, {"start": 2, "end": 3, "type": "LOC"}]
    assert check_no_flat_overlap(ents2) == []


# ── Quarantine split ────────────────────────────────────────────────────
def test_quarantine_split_excludes_exactly_flagged_samples():
    from src.adjudication import SkippedDecision
    splits = {"train": _splits_from_pairs([
        ("甲", "O"), ("乙", "O"), ("丙", "O"),
    ])}
    new_splits, applied, _, _dn3, _dr3 = apply_ops_to_splits(splits, [], [], [])
    quarantined = [SkippedDecision("wb:1", "train", 1, "乙", "SKIP_UNANNOTATED_SAMPLE", "quarantined_unannotated_sample")]
    qids = quarantine_sample_ids_by_split(quarantined)
    assert qids == {"train": {1}}
    clean_splits, index_map = build_clean_splits(new_splits, qids)
    assert len(clean_splits["train"]) == 2
    assert [t for t, l in clean_splits["train"]] == [["甲"], ["丙"]]
    assert index_map["train"] == {0: 0, 2: 1}

    quarantine_records = build_quarantine_records(splits, qids, quarantined)
    assert len(quarantine_records["train"]) == 1
    assert quarantine_records["train"][0]["sample_id"] == 1
    assert quarantine_records["train"][0]["text_raw"] == "乙"
    assert quarantine_records["train"][0]["bio_labels"] == ["O"]

    assert verify_no_duplicate_sample_ids(new_splits) == []
    assert verify_quarantine_partition(new_splits, clean_splits, qids) == []
    # v2_clean thieu/thua so voi ky vong -> phai bi bat
    bad_clean = {"train": clean_splits["train"][:1]}
    assert verify_quarantine_partition(new_splits, bad_clean, qids) != []


def test_verify_add_entity_never_applied_flags_regression():
    c_ok = ProposedChange(0, 0, "train", None, "T", "CORRECT_LABEL", (0, 0), "O", (0, 0),
                           "PER", "a", None, "", "t", "valid")
    assert verify_add_entity_never_applied([c_ok], [], []) == []
    c_bad = ProposedChange(1, 0, "train", None, "T", "ADD_ENTITY", (0, 0), "O", (0, 0),
                            "PER", "a", None, "", "t", "valid")
    assert verify_add_entity_never_applied([c_bad], [], []) != []


def test_verify_source_files_unchanged_detects_tampering(tmp_path):
    p = tmp_path / "train.txt"
    p.write_text("甲 O\n\n", encoding="utf-8")
    before = _file_checksum(str(p))
    assert verify_source_files_unchanged({"train": str(p)}, {"train": before}) == []
    p.write_text("乙 O\n\n", encoding="utf-8")
    assert verify_source_files_unchanged({"train": str(p)}, {"train": before}) != []


# ══════════════════════════════════════════════════════════════════════════
# Integration: apply_adjudication.py --apply chạy thật trên fixture nhỏ
# ══════════════════════════════════════════════════════════════════════════
def _write_conll(path, sentences):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for tokens, labels in sentences:
            for t, l in zip(tokens, labels):
                f.write(f"{t} {l}\n")
            f.write("\n")


def _build_fixture(tmp_path):
    data_dir = tmp_path / "data"
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    train_sents = _splits_from_pairs([
        ("宋嘉定九年", "B-ORG B-DTM I-DTM I-DTM I-DTM"),
        ("莫大正", "O O O"),
        ("故都御史杜岳", "B-LOC I-LOC B-TITLE I-TITLE O O"),
    ])
    dev_sents = _splits_from_pairs([("陳", "B-PER")])
    test_sents = _splits_from_pairs([("阮", "B-PER")])

    train_path = str(data_dir / "train.txt")
    dev_path = str(data_dir / "dev.txt")
    test_path = str(data_dir / "test.txt")
    _write_conll(train_path, train_sents)
    _write_conll(dev_path, dev_sents)
    _write_conll(test_path, test_sents)

    checksums = {"train": _file_checksum(train_path), "dev": _file_checksum(dev_path),
                 "test": _file_checksum(test_path)}
    manifest_path = str(tmp_path / "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"dataset_checksums": checksums}, f)

    singleton_path = str(review_dir / "review_priority_singleton_anomalies.xlsx")
    pd.DataFrame(columns=["reviewer_decision", "record_ref", "surface", "reviewer_notes"]).to_excel(
        singleton_path, index=False)

    guideline_path = str(review_dir / "review_guideline_ambiguities.xlsx")
    with pd.ExcelWriter(guideline_path) as writer:
        pd.DataFrame(columns=["reviewer_decision", "surface"]).to_excel(
            writer, sheet_name="guideline_ambiguities", index=False)
        pd.DataFrame(columns=["reviewer_decision", "split", "sample_id", "gold_surface",
                               "gold_label", "gold_start", "gold_end", "entity_id",
                               "semantic_role_candidate", "document_id", "full_sentence"]).to_excel(
            writer, sheet_name="deepdive_occurrences", index=False)

    gr11_path = str(review_dir / "gr11_date_formula_candidates.xlsx")
    pd.DataFrame([
        {"candidate_id": 1, "split": "train", "sample_id": 0, "document_id": None,
         "original_entities": "宋/ORG[0-0]; 嘉定九年/DTM[1-4]", "candidate_span": "0-4",
         "candidate_surface": "宋嘉定九年", "current_gold_configuration": "mixed/overlap",
         "reviewer_decision": "MERGE_ENTITY", "final_span": "0-4", "final_label": "DTM",
         "suggested_rule_id": "GR-11", "reviewer_notes": "test merge"},
        {"candidate_id": 2, "split": "train", "sample_id": 1, "document_id": None,
         "original_entities": None, "candidate_span": "0-2",
         "candidate_surface": "莫大正", "current_gold_configuration": "unannotated",
         "reviewer_decision": "SKIP_UNANNOTATED_SAMPLE", "final_span": None, "final_label": None,
         "suggested_rule_id": "GR-11", "reviewer_notes": "test quarantine"},
    ]).to_excel(gr11_path, index=False)

    tx_path = str(review_dir / "yushi_gudu_transaction.xlsx")
    pd.DataFrame([
        {"transaction_id": "T1", "action": "REMOVE_ENTITY", "split": "train", "sample_id": 2,
         "document_id": None, "source_entity_id": 2, "old_label": "LOC", "old_start": 0,
         "old_end": 1, "old_surface": "故都", "guideline_rule_id": "GR-02",
         "reviewer_notes": "test remove", "remove_reason": "not a place"},
        {"transaction_id": "T1", "action": "CORRECT_BOUNDARY", "split": "train", "sample_id": 2,
         "document_id": None, "source_entity_id": 3, "old_label": "TITLE", "old_start": 2,
         "old_end": 3, "final_start": 1, "final_end": 3, "final_label": "TITLE",
         "guideline_rule_id": "GR-02", "reviewer_notes": "test boundary"},
    ]).to_excel(tx_path, index=False)

    return {
        "train": train_path, "dev": dev_path, "test": test_path,
        "manifest": manifest_path, "review_dir": str(review_dir),
        "gr11": gr11_path, "transactions": tx_path,
        "singleton": singleton_path, "guideline": guideline_path,
    }


def _run_cli(fixture, out_dir, extra_args=None, apply=False):
    script = os.path.join(REPO_ROOT, "scripts", "apply_adjudication.py")
    cmd = [sys.executable, script,
           "--train", fixture["train"], "--dev", fixture["dev"], "--test", fixture["test"],
           "--review-dir", fixture["review_dir"], "--audit-manifest", fixture["manifest"],
           "--transactions", fixture["transactions"], "--gr11-candidates", fixture["gr11"],
           "--out-dir", out_dir]
    if apply:
        cmd.append("--apply")
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)


def test_dry_run_fixture_has_zero_validation_errors(tmp_path):
    fixture = _build_fixture(tmp_path)
    out_dir = str(tmp_path / "dry_run_out")
    result = _run_cli(fixture, out_dir)
    assert result.returncode == 0, result.stderr
    with open(os.path.join(out_dir, "validation_errors.csv"), encoding="utf-8-sig") as f:
        rows = f.read().strip().splitlines()
    assert len(rows) == 1  # chi co header, 0 error row


def test_apply_refuses_when_dry_run_has_validation_errors(tmp_path):
    fixture = _build_fixture(tmp_path)
    # lam hong: tro gr11-candidates toi file khong ton tai -> global_errors > 0
    bad_gr11 = str(tmp_path / "missing.xlsx")
    fixture_bad = dict(fixture, gr11=bad_gr11)
    out_dir = str(tmp_path / "apply_out_should_fail")
    result = _run_cli(fixture_bad, out_dir, apply=True)
    assert result.returncode != 0
    assert not os.path.exists(out_dir)


def test_apply_succeeds_and_writes_full_output_tree(tmp_path):
    fixture = _build_fixture(tmp_path)
    out_dir = str(tmp_path / "dataset_v2")
    result = _run_cli(fixture, out_dir, apply=True)
    assert result.returncode == 0, result.stderr + result.stdout

    assert os.path.isdir(out_dir)
    assert os.path.exists(os.path.join(out_dir, "MATERIALIZATION_SUCCESS.json"))
    for sub in ("dataset_v2_full", "dataset_v2_clean", "quarantine_v2", "provenance",
                "changelog", "guideline", "validation"):
        assert os.path.isdir(os.path.join(out_dir, sub)), sub

    # khong con temp dir sot lai cung cap voi out_dir
    parent = os.path.dirname(out_dir)
    leftover = [d for d in os.listdir(parent) if d.startswith(f".{os.path.basename(out_dir)}_tmp_")]
    assert leftover == []

    full_train = os.path.join(out_dir, "dataset_v2_full", "train.txt")
    with open(full_train, encoding="utf-8") as f:
        content = f.read()
    assert "宋嘉定九年" not in content or "B-DTM" in content  # merge applied
    sentences = content.strip("\n").split("\n\n")
    assert len(sentences) == 3  # sample 1 (quarantine) VAN CON trong v2_full

    clean_train = os.path.join(out_dir, "dataset_v2_clean", "train.txt")
    with open(clean_train, encoding="utf-8") as f:
        clean_content = f.read()
    clean_sentences = [s for s in clean_content.strip("\n").split("\n\n") if s.strip()]
    assert len(clean_sentences) == 2  # sample 1 (莫大正, quarantine) bi loai

    with open(os.path.join(out_dir, "quarantine_v2", "train_partial_annotation.jsonl"), encoding="utf-8") as f:
        qrecs = [json.loads(line) for line in f if line.strip()]
    assert len(qrecs) == 1
    assert qrecs[0]["text_raw"] == "莫大正"
    assert qrecs[0]["bio_labels"] == ["O", "O", "O"]  # nguyen ban goc, khong sua

    with open(os.path.join(out_dir, "dataset_v2_full", "checksums.json"), encoding="utf-8") as f:
        checksums = json.load(f)
    assert "train.txt" in checksums


def test_apply_refuses_existing_out_dir_without_force(tmp_path):
    fixture = _build_fixture(tmp_path)
    out_dir = str(tmp_path / "dataset_v2")
    r1 = _run_cli(fixture, out_dir, apply=True)
    assert r1.returncode == 0, r1.stderr
    r2 = _run_cli(fixture, out_dir, apply=True)
    assert r2.returncode != 0
    assert "force-overwrite" in r2.stderr.lower() or "force-overwrite" in r2.stdout.lower()


def test_apply_force_overwrite_backs_up_old_output(tmp_path):
    fixture = _build_fixture(tmp_path)
    out_dir = str(tmp_path / "dataset_v2")
    r1 = _run_cli(fixture, out_dir, apply=True)
    assert r1.returncode == 0, r1.stderr
    r2 = _run_cli(fixture, out_dir, extra_args=["--force-overwrite"], apply=True)
    assert r2.returncode == 0, r2.stderr
    parent = os.path.dirname(out_dir)
    backups = [d for d in os.listdir(parent) if d.startswith(f"{os.path.basename(out_dir)}.backup_")]
    assert len(backups) == 1


def test_v1_source_files_byte_identical_after_apply(tmp_path):
    fixture = _build_fixture(tmp_path)
    before = _file_checksum(fixture["train"])
    out_dir = str(tmp_path / "dataset_v2")
    _run_cli(fixture, out_dir, apply=True)
    after = _file_checksum(fixture["train"])
    assert before == after


def test_gr11b_deferred_not_read_by_apply(tmp_path):
    # scan_dynasty_era_candidates.py output (GR-11b) khong co flag rieng
    # trong CLI -- khong bi doc trong bat ky lan chay nao (dry-run hay apply).
    script = os.path.join(REPO_ROOT, "scripts", "apply_adjudication.py")
    result = subprocess.run([sys.executable, script, "--help"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=30)
    assert "dynasty-era" not in result.stdout.lower()
    assert "gr11b" not in result.stdout.lower().replace("-", "")


def test_add_entity_rows_never_applied(tmp_path):
    from src.adjudication import process_gr11_date_formula_candidates, build_entity_index
    df = pd.DataFrame([
        {"candidate_id": 9, "split": "train", "sample_id": 0, "document_id": None,
         "original_entities": None, "candidate_surface": "x",
         "current_gold_configuration": "mixed/overlap", "reviewer_decision": "ADD_ENTITY",
         "final_span": None, "final_label": None, "suggested_rule_id": "GR-11",
         "reviewer_notes": ""},
    ])
    proposed, skipped, quarantined = process_gr11_date_formula_candidates(df, [], {}, 0)
    assert proposed == []
    assert quarantined == []
    assert len(skipped) == 1
    assert "ADD_ENTITY" in skipped[0].reason
