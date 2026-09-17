import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_results import load_experiment, paired_deltas, summarize


def _write_result(path, seed, test_f1, per_f1=0.7, start_f1=None, end_f1=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "seed": seed,
        "best_epoch": 3,
        "best_dev_f1": test_f1 + 0.01,
        "test_f1": test_f1,
        "test_report": {
            "micro avg": {"precision": test_f1, "recall": test_f1, "f1-score": test_f1},
            "PER": {"f1-score": per_f1, "precision": per_f1, "recall": per_f1, "support": 10},
            "LOC": {"f1-score": 0.6, "precision": 0.6, "recall": 0.6, "support": 8},
        },
        "test_start_f1": start_f1,
        "test_end_f1": end_f1,
        "total_params": 12345,
        "train_time": 100.0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_load_experiment_and_summarize(tmp_path):
    for seed, f1 in [(42, 0.60), (43, 0.62), (44, 0.58)]:
        _write_result(str(tmp_path / f"m0_s{seed}" / "results.json"), seed, f1)

    rows = load_experiment("M0", str(tmp_path / "m0_s*" / "results.json"))
    assert len(rows) == 3
    assert {r["seed"] for r in rows} == {42, 43, 44}

    summary = summarize(rows)
    assert summary["n_seeds"] == 3
    assert abs(summary["test_f1_mean"] - 0.6) < 1e-6
    assert summary["test_f1_min"] == 0.58
    assert summary["test_f1_max"] == 0.62


def test_load_experiment_duplicate_seed_raises(tmp_path):
    _write_result(str(tmp_path / "a" / "results.json"), 42, 0.5)
    _write_result(str(tmp_path / "b" / "results.json"), 42, 0.6)
    try:
        load_experiment("M0", str(tmp_path / "*" / "results.json"))
        assert False, "expected ValueError on duplicate seed"
    except ValueError:
        pass


def test_paired_deltas_matches_common_seeds_only():
    rows_a = [{"seed": 42, "test_f1": 0.60}, {"seed": 43, "test_f1": 0.62}]
    rows_b = [{"seed": 42, "test_f1": 0.65}, {"seed": 44, "test_f1": 0.70}]
    deltas = paired_deltas(rows_a, rows_b, metric="test_f1")
    assert deltas == {42: 0.05}
