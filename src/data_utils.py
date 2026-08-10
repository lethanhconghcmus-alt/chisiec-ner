"""
data_utils.py — CoNLL reader, validation, NERDataset, DataLoader factory
"""

import os
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.utils import get_logger
from src.gazetteer import GAZ_LABEL2ID

logger = get_logger(__name__)


# ── CONLL READER ──────────────────────────────────────────────────────────────
def read_conll(path: str) -> list:
    """
    Đọc file CoNLL BIO format.
    Mỗi dòng: 'token label' (hoặc nhiều cột, lấy cột đầu & cuối).
    Câu cách nhau bằng dòng trống.
    Returns: list of (tokens, labels)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    sentences, tokens, labels = [], [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip()
            if line == "" or line.startswith("-DOCSTART-"):
                if tokens:
                    sentences.append((tokens[:], labels[:]))
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Line {line_no} malformed (skipped): '{line}'")
                    continue
                tokens.append(parts[0])
                labels.append(parts[-1])
    if tokens:
        sentences.append((tokens, labels))

    logger.info(f"Loaded {len(sentences)} sentences from {path}")
    return sentences


# ── DATA VALIDATION ───────────────────────────────────────────────────────────
def validate_data(data: list, split_name: str = "data") -> dict:
    """
    Kiểm tra data: nhãn hợp lệ, câu trống, token lạ.
    Returns: dict thống kê
    """
    errors, warnings_list = [], []
    label_counter = Counter()
    token_lengths = []
    bio_errors    = 0

    for i, (tokens, labels) in enumerate(data):
        if len(tokens) == 0:
            errors.append(f"Sentence {i}: empty")
            continue
        if len(tokens) != len(labels):
            errors.append(
                f"Sentence {i}: token/label length mismatch "
                f"({len(tokens)} vs {len(labels)})"
            )

        label_counter.update(labels)
        token_lengths.append(len(tokens))

        # Kiểm tra BIO consistency
        prev_label = "O"
        for j, label in enumerate(labels):
            if label.startswith("I-"):
                entity = label[2:]
                if prev_label not in (f"B-{entity}", f"I-{entity}"):
                    bio_errors += 1
            prev_label = label

    stats = {
        "total_sentences": len(data),
        "label_distribution": dict(label_counter.most_common()),
        "avg_length": round(sum(token_lengths) / max(len(token_lengths), 1), 2),
        "max_length": max(token_lengths) if token_lengths else 0,
        "min_length": min(token_lengths) if token_lengths else 0,
        "bio_errors": bio_errors,
        "errors": errors,
    }

    if errors:
        logger.error(f"[{split_name}] {len(errors)} errors found!")
        for e in errors[:5]:
            logger.error(f"  → {e}")
    if bio_errors:
        logger.warning(f"[{split_name}] {bio_errors} BIO inconsistencies")
    else:
        logger.info(f"[{split_name}] Validation passed ✅")

    return stats


# ── LABEL MAP ─────────────────────────────────────────────────────────────────
def build_label_map(train_data: list) -> tuple:
    label_set = sorted({l for _, labs in train_data for l in labs})
    label2id  = {l: i for i, l in enumerate(label_set)}
    id2label  = {i: l for l, i in label2id.items()}
    logger.info(f"Labels ({len(label2id)}): {label_set}")
    return label2id, id2label


# ── DATASET ───────────────────────────────────────────────────────────────────
class NERDataset(Dataset):

    def __init__(self, data, tokenizer, label2id, max_len=128, gaz_tagger=None):
        self.data       = data
        self.tok        = tokenizer
        self.label2id   = label2id
        self.max_len    = max_len
        self.gaz_tagger = gaz_tagger

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, labels = self.data[idx]

        enc = self.tok(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        gaz_tags = self.gaz_tagger.tag(tokens) if self.gaz_tagger is not None else None

        label_ids = []
        gaz_ids   = []
        prev_word = None

        for word_id in enc.word_ids():
            if word_id is None:
                label_ids.append(-100)
                gaz_ids.append(0)
            elif word_id != prev_word:
                label_ids.append(self.label2id[labels[word_id]])
                if gaz_tags is not None:
                    gaz_ids.append(GAZ_LABEL2ID[gaz_tags[word_id]])
                else:
                    gaz_ids.append(0)
            else:
                label_ids.append(-100)
                gaz_ids.append(0)
            prev_word = word_id

        item = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
            "labels":         torch.tensor(label_ids),
        }
        if self.gaz_tagger is not None:
            item["gaz_ids"] = torch.tensor(gaz_ids)
        return item


# ── DATALOADER FACTORY ────────────────────────────────────────────────────────
def make_dataloader(
    data:        list,
    tokenizer:   AutoTokenizer,
    label2id:    dict,
    max_len:     int,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int = 0,
    gaz_tagger=None,
) -> DataLoader:
    dataset = NERDataset(data, tokenizer, label2id, max_len, gaz_tagger=gaz_tagger)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
