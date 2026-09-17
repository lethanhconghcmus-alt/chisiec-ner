import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from src.bioes_utils import build_bioes_label_map
from src.data_utils import NERDataset

TINY_BACKBONE = "hf-internal-testing/tiny-random-bert"


def _tok():
    return AutoTokenizer.from_pretrained(TINY_BACKBONE)


def test_boundary_labels_derived_and_aligned_to_first_subword():
    tokenizer = _tok()
    label2id, _ = build_bioes_label_map()

    tokens = ["太", "師", "陳", "守", "度", "至", "化", "州"]
    labels = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "B-LOC", "E-LOC"]
    data = [(tokens, labels)]

    ds = NERDataset(data, tokenizer, label2id, max_len=16, derive_boundary=True)
    item = ds[0]

    assert "start_labels" in item and "end_labels" in item
    assert item["start_labels"].shape == item["labels"].shape
    assert item["end_labels"].shape == item["labels"].shape

    # mọi vị trí có labels == -100 (CLS/SEP/pad) phải có start/end == -100
    ignore_pos = (item["labels"] == -100)
    assert torch.all(item["start_labels"][ignore_pos] == -100)
    assert torch.all(item["end_labels"][ignore_pos] == -100)

    # mọi vị trí có labels != -100 phải có start/end trong {0, 1}
    valid_pos = ~ignore_pos
    assert torch.all((item["start_labels"][valid_pos] == 0) | (item["start_labels"][valid_pos] == 1))
    assert torch.all((item["end_labels"][valid_pos] == 0) | (item["end_labels"][valid_pos] == 1))


def test_boundary_disabled_by_default_no_extra_keys():
    tokenizer = _tok()
    label2id, _ = build_bioes_label_map()
    data = [(["太", "師"], ["B-TITLE", "E-TITLE"])]
    ds = NERDataset(data, tokenizer, label2id, max_len=8)
    item = ds[0]
    assert "start_labels" not in item
    assert "end_labels" not in item


def test_start_end_values_match_derive_boundary_labels_for_word_positions():
    tokenizer = _tok()
    label2id, _ = build_bioes_label_map()
    tokens = ["太", "師", "陳", "守", "度", "至", "化", "州"]
    labels = ["B-TITLE", "E-TITLE", "B-PER", "I-PER", "E-PER", "O", "B-LOC", "E-LOC"]
    expected_start = [1, 0, 1, 0, 0, 0, 1, 0]
    expected_end   = [0, 1, 0, 0, 1, 0, 0, 1]

    ds = NERDataset([(tokens, labels)], tokenizer, label2id, max_len=16, derive_boundary=True)
    item = ds[0]

    # với tiny-random-bert, mỗi ký tự Hán có thể bị tokenize thành [UNK] (1
    # subword/word vẫn giữ đúng 1-1 vì input đã is_split_into_words=True) —
    # lấy đúng các vị trí ứng với mỗi word_id qua encode lại để so khớp.
    enc = tokenizer(tokens, is_split_into_words=True, max_length=16,
                     padding="max_length", truncation=True)
    word_ids = enc.word_ids()
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        assert item["start_labels"][pos].item() == expected_start[wid]
        assert item["end_labels"][pos].item() == expected_end[wid]
