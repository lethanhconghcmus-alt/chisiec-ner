import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator_extended import build_seen_surfaces, extended_test_report


def test_build_seen_surfaces_from_train_entities():
    train_data = [
        (list("宋嘉定九年"), "B-ORG B-DTM I-DTM I-DTM I-DTM".split()),
    ]
    seen = build_seen_surfaces(train_data)
    assert ("宋", "ORG") in seen
    assert ("嘉定九年", "DTM") in seen
    assert ("嘉", "DTM") not in seen


def test_extended_report_handles_truncated_sentence_without_crash(tmp_path):
    # cau goc dai 6 ky tu, nhung "infer" (mo phong max_len cat) chi tra ve
    # 4 vi tri dau -- extended_test_report PHAI tu align lai tokens, KHONG
    # duoc IndexError, va phai dem n_sentences_with_subword_coverage_gap=1.
    full_tokens = list("明成化十七年")  # 6 ky tu
    full_labels = "B-DTM I-DTM I-DTM I-DTM I-DTM I-DTM".split()
    raw_data = [(full_tokens, full_labels)]
    train_data = [(full_tokens, full_labels)]

    truncated_gold = full_labels[:4]
    truncated_pred = ["B-DTM", "I-DTM", "I-DTM", "I-DTM"]

    with patch("src.evaluator_extended._run_inference",
               return_value=[(truncated_gold, truncated_pred, None)]):
        report = extended_test_report(
            model=object(), id2label={}, device="cpu", output_dir=str(tmp_path),
            loader=None, raw_data=raw_data, train_data=train_data, split="test",
        )

    assert report["n_sentences_with_subword_coverage_gap"] == 1
    assert os.path.exists(tmp_path / "test_extended_report.json")
    assert os.path.exists(tmp_path / "test_predictions_full.json")


def test_extended_report_no_truncation_reports_zero(tmp_path):
    full_tokens = list("宋嘉定九年")
    full_labels = "B-ORG B-DTM I-DTM I-DTM I-DTM".split()
    raw_data = [(full_tokens, full_labels)]
    train_data = [(full_tokens, full_labels)]

    with patch("src.evaluator_extended._run_inference", return_value=[(full_labels, full_labels, None)]):
        report = extended_test_report(
            model=object(), id2label={}, device="cpu", output_dir=str(tmp_path),
            loader=None, raw_data=raw_data, train_data=train_data, split="test",
        )

    assert report["n_sentences_with_subword_coverage_gap"] == 0
    assert report["span_category_counts"].get("exact_span_correct_type") == 1


def test_extended_report_uses_constrained_tags_when_provided(tmp_path):
    full_tokens = list("宋嘉定九年")
    full_labels = "B-ORG B-DTM I-DTM I-DTM I-DTM".split()
    raw_data = [(full_tokens, full_labels)]
    train_data = [(full_tokens, full_labels)]
    # raw decode co "loi" (khac gold), constrained decode "dung" (khop gold)
    # -- extended_test_report PHAI dung constrained cho entity extraction.
    raw_wrong = "O O O O O".split()
    constrained_correct = full_labels

    with patch("src.evaluator_extended._run_inference",
               return_value=[(full_labels, raw_wrong, constrained_correct)]):
        report = extended_test_report(
            model=object(), id2label={}, device="cpu", output_dir=str(tmp_path),
            loader=None, raw_data=raw_data, train_data=train_data, split="test",
            constrained_crf=object(),
        )

    assert report["decode_policy"] == "constrained"
    # _categorize_spans tra ve 1 SET category/cau (khong dem tung entity) --
    # 1 cau voi ca 2 entity khop het chi tao "exact_span_correct_type" 1 lan.
    assert report["span_category_counts"] == {"exact_span_correct_type": 1}
    assert report["bio_violations"]["constrained_total_violations"] == 0


def test_extended_report_raises_on_sentence_count_mismatch(tmp_path):
    raw_data = [(list("甲"), ["O"]), (list("乙"), ["O"])]
    with patch("src.evaluator_extended._run_inference", return_value=[(["O"], ["O"], None)]):
        try:
            extended_test_report(
                model=object(), id2label={}, device="cpu", output_dir=str(tmp_path),
                loader=None, raw_data=raw_data, train_data=raw_data, split="test",
            )
            assert False, "phai raise ValueError khi so cau infer khac raw_data"
        except ValueError:
            pass
