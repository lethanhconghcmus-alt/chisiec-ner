"""
Predictor — loads model once at startup and exposes predict() / predict_document().
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
CKPT_PATH  = os.getenv("CHECKPOINT_PATH",  "outputs/ancient/guwenbert_crf/best.pt")
LABEL_MAP  = os.getenv("LABEL_MAP_PATH", os.getenv("LABEL_MAP", "outputs/ancient/guwenbert_crf/label_map.json"))
MAX_LEN    = int(os.getenv("MAX_LEN", "128"))
DEVICE     = torch.device(os.getenv("DEVICE",  "cpu"))


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

    def predict_document(self, sentences: List[str]) -> List[List[Dict]]:
        """
        Predict NER entities for a list of consecutive sentences.

        Sentences are joined and re-cut into windows of <= MAX_LEN - 2 chars
        (preferring sentence-final marks, then commas) so that short sentences
        keep their context and long ones are not truncated. Entities are then
        mapped back to per-sentence offsets.
        """
        full = "".join(sentences)
        max_chars = MAX_LEN - 2

        doc_entities: List[Dict] = []
        start = 0
        while start < len(full):
            end = min(start + max_chars, len(full))
            if end < len(full):
                cut = max(full.rfind(p, start, end) for p in "。？！；?!;")
                if cut <= start:
                    cut = max(full.rfind(p, start, end) for p in "，、：,:")
                if cut > start:
                    end = cut + 1
            for ent in self.predict(full[start:end]):
                doc_entities.append({**ent, "start": ent["start"] + start, "end": ent["end"] + start})
            start = end

        results: List[List[Dict]] = []
        offset = 0
        for sent in sentences:
            sent_end = offset + len(sent)
            ents = []
            for ent in doc_entities:
                if offset <= ent["start"] < sent_end:
                    s, e = ent["start"] - offset, min(ent["end"], sent_end) - offset
                    ents.append({"text": sent[s:e], "label": ent["label"], "start": s, "end": e})
            results.append(ents)
            offset = sent_end
        return results
