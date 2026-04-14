"""
Predictor — loads model once at startup and exposes predict().
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from src.models import GuwenBertCRF

logger = logging.getLogger("api.predictor")

BACKBONE   = os.getenv("BACKBONE",   "ethanyt/guwenbert-base")
CKPT_PATH  = os.getenv("CKPT_PATH",  "outputs/ancient/guwenbert_crf/best.pt")
LABEL_MAP  = os.getenv("LABEL_MAP",  "outputs/ancient/guwenbert_crf/label_map.json")
MAX_LEN    = int(os.getenv("MAX_LEN", "128"))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    def __init__(self) -> None:
        logger.info("Loading tokenizer from %s", BACKBONE)
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

        logger.info("Loading label map from %s", LABEL_MAP)
        with open(LABEL_MAP, encoding="utf-8") as f:
            maps = json.load(f)
        self.label2id: Dict[str, int] = maps["label2id"]
        self.id2label: Dict[int, str] = {int(k): v for k, v in maps["id2label"].items()}
        num_labels = len(self.label2id)

        logger.info("Loading model checkpoint from %s", CKPT_PATH)
        self.model = GuwenBertCRF(BACKBONE, num_labels)
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()
        logger.info("Model ready on %s | labels=%d", DEVICE, num_labels)

    @torch.no_grad()
    def predict(self, sentence: str) -> List[Dict]:
        """
        Predict NER entities for a single sentence string.

        Returns a list of entity dicts:
          {"text": str, "label": str, "start": int, "end": int}
        """
        tokens = list(sentence.strip())
        if not tokens:
            return []

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        token_type_ids = encoding.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        pred_ids = self.model(input_ids, attention_mask, token_type_ids)
        pred_ids = pred_ids[0]  # batch size = 1

        word_ids = encoding.word_ids(batch_index=0)
        token_labels: List[str] = []
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None:
                continue
            if len(token_labels) <= wid:
                token_labels.append(self.id2label[pid])

        # BIO → entity spans
        entities = []
        cur_tokens: List[str] = []
        cur_label: str | None = None
        cur_start: int = 0

        for i, (tok, label) in enumerate(zip(tokens[:len(token_labels)], token_labels)):
            if label.startswith("B-"):
                if cur_tokens:
                    entities.append({
                        "text":  "".join(cur_tokens),
                        "label": cur_label,
                        "start": cur_start,
                        "end":   cur_start + len(cur_tokens),
                    })
                cur_tokens = [tok]
                cur_label  = label[2:]
                cur_start  = i
            elif label.startswith("I-") and cur_label == label[2:]:
                cur_tokens.append(tok)
            else:
                if cur_tokens:
                    entities.append({
                        "text":  "".join(cur_tokens),
                        "label": cur_label,
                        "start": cur_start,
                        "end":   cur_start + len(cur_tokens),
                    })
                cur_tokens = []
                cur_label  = None

        if cur_tokens:
            entities.append({
                "text":  "".join(cur_tokens),
                "label": cur_label,
                "start": cur_start,
                "end":   cur_start + len(cur_tokens),
            })

        return entities
