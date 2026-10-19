"""
Predictor — NER models (BERT encoder + CRF) selectable by name.

Each model is loaded once, on first use, and exposes predict() / predict_document().
Select the default with NER_MODEL; a request can also ask for any model in
available_models() (= registered models whose checkpoint file exists).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from src.models import GuwenBertCRF as BertCRF  # generic encoder + Linear + CRF

logger = logging.getLogger("api.predictor")

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
LABEL_MAP      = os.getenv("LABEL_MAP_PATH", os.getenv("LABEL_MAP", "artifacts/label_map.json"))
DEVICE         = torch.device(os.getenv("DEVICE", "cpu"))

# All models share the 11-label BIO map in artifacts/label_map.json (PER/LOC/ORG/DTM/TITLE).
# F1 = entity-level exact match on each model's own test set.
NER_MODELS: Dict[str, Dict] = {
    "guwenbert-crf-tq": {
        "backbone": "ethanyt/guwenbert-base",
        "checkpoint": os.getenv("CHECKPOINT_PATH", os.path.join(CHECKPOINT_DIR, "best.pt")),
        "max_len": 128,
        "constrained": False,
        "punct_to": None,
        "description": "GuwenBERT + CRF, trained on 4 Chinese histories (AnChineseNERE, CHisIEC, "
                       "BEEVETA, C-CLUE). Test F1 0.932 on that data; not trained on DVSKTT.",
    },
    "sikubert-crf-dvsktt": {
        "backbone": "SIKU-BERT/sikubert",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "sikubert_crf_dvsktt.pt"),
        "max_len": 256,
        "constrained": True,
        "punct_to": ".",
        "description": "SikuBERT + CRF, fine-tuned on DVSKTT dataset_v2_clean "
                       "(benchmark v2 R2, seed 44). DVSKTT test F1 0.678.",
    },
    "gujiroberta-crf-dvsktt": {
        "backbone": "hsc748NLP/GujiRoBERTa_jian",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "gujiroberta_crf_dvsktt.pt"),
        "max_len": 256,
        "constrained": True,
        "punct_to": ".",
        "description": "GujiRoBERTa (jian) + CRF, fine-tuned on DVSKTT dataset_v1. "
                       "DVSKTT test F1 0.663.",
    },
}
DEFAULT_MODEL = os.getenv("NER_MODEL", "guwenbert-crf-tq")

# Punctuation never belongs to an entity. DVSKTT training text separates sentences with
# ASCII "." and has almost no CJK marks, so DVSKTT models get CJK marks rewritten to "."
# ("punct_to") -- one char for one char, so offsets are unchanged.
CJK_PUNCT = set("，。：；？！、")
PUNCT = CJK_PUNCT | set(".,:;?!“”‘’「」『』（）《》〈〉【】…—·\"'()[]")


def available_models() -> List[str]:
    return [name for name, cfg in NER_MODELS.items() if os.path.exists(cfg["checkpoint"])]


def _constrain_bio(crf, id2label: Dict[int, str]) -> None:
    """Forbid illegal BIO transitions in the CRF (I-X only after B-X / I-X, never at start)."""
    big_neg = -1e4
    with torch.no_grad():
        for j, to in id2label.items():
            if not to.startswith("I-"):
                continue
            crf.start_transitions[j] = big_neg
            for i, frm in id2label.items():
                if frm[2:] != to[2:] or frm == "O":
                    crf.transitions[i, j] = big_neg


class Predictor:
    def __init__(self, name: str = DEFAULT_MODEL) -> None:
        if name not in NER_MODELS:
            raise ValueError(f"Unknown NER model '{name}'. Known: {list(NER_MODELS)}")
        cfg = NER_MODELS[name]
        self.name = name
        self.max_len: int = cfg["max_len"]
        self.punct_to: str | None = cfg["punct_to"]

        logger.info("[%s] Loading tokenizer from %s", name, cfg["backbone"])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])

        with open(LABEL_MAP, encoding="utf-8") as f:
            maps = json.load(f)
        self.id2label: Dict[int, str] = {int(k): v for k, v in maps["id2label"].items()}

        logger.info("[%s] Loading checkpoint from %s", name, cfg["checkpoint"])
        self.model = BertCRF(cfg["backbone"], len(self.id2label))
        self.model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
        if cfg["constrained"]:
            _constrain_bio(self.model.crf, self.id2label)
        self.model.to(DEVICE)
        self.model.eval()
        logger.info("[%s] Model ready on %s", name, DEVICE)

    @torch.no_grad()
    def tag(self, texts: List[str]) -> List[List[str]]:
        """BIO label per character for a batch of texts (each truncated to max_len tokens)."""
        model_texts = [[self.punct_to if self.punct_to and c in CJK_PUNCT else c for c in t]
                       for t in texts]
        encoding = self.tokenizer(
            model_texts,
            is_split_into_words=True,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        token_type_ids = encoding.get("token_type_ids")
        pred_ids = self.model(
            encoding["input_ids"].to(DEVICE),
            encoding["attention_mask"].to(DEVICE),
            token_type_ids.to(DEVICE) if token_type_ids is not None else None,
        )

        # Label each character by its first sub-token. Characters the tokenizer
        # drops entirely (e.g. Private Use Area code points, frequent in DVSKTT)
        # get no token and stay "O" -- indexing by word id keeps later labels aligned.
        result = []
        for b, text in enumerate(texts):
            labels = ["O"] * len(text)
            seen = set()
            for tok_idx, wid in enumerate(encoding.word_ids(batch_index=b)):
                if wid is not None and wid not in seen and tok_idx < len(pred_ids[b]):
                    seen.add(wid)
                    if text[wid] not in PUNCT:
                        labels[wid] = self.id2label[pred_ids[b][tok_idx]]
            result.append(labels)
        return result

    def predict(self, sentence: str) -> List[Dict]:
        """
        Predict NER entities for a single sentence (truncated to max_len tokens).

        Returns a list of entity dicts:
          {"text": str, "label": str, "start": int, "end": int}
        """
        tokens = list(sentence.strip())
        if not tokens:
            return []
        labels = self.tag(["".join(tokens)])[0]

        # BIO → entity spans
        entities: List[Dict] = []
        cur_label: str | None = None
        cur_start = 0
        for i, label in enumerate(labels + ["O"]):
            if label.startswith("I-") and cur_label == label[2:]:
                continue
            if cur_label is not None:
                entities.append({"text": "".join(tokens[cur_start:i]), "label": cur_label,
                                 "start": cur_start, "end": i})
                cur_label = None
            if label != "O":  # B-X, or I-X without a matching open entity
                cur_label, cur_start = label[2:], i
        return entities

    def predict_document(self, sentences: List[str]) -> List[List[Dict]]:
        """
        Predict NER entities for a list of consecutive sentences.

        Sentences are joined and re-cut into windows of <= max_len - 2 chars
        (preferring sentence-final marks, then commas) so that short sentences
        keep their context and long ones are not truncated. Entities are then
        mapped back to per-sentence offsets.
        """
        full = "".join(sentences)
        max_chars = self.max_len - 2

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


_predictors: Dict[str, Predictor] = {}
_lock = threading.Lock()


def get_predictor(name: str | None = None) -> Predictor:
    """Return the predictor for `name` (default model if None), loading it on first use."""
    name = name or DEFAULT_MODEL
    if name not in available_models():
        raise KeyError(name)
    with _lock:
        if name not in _predictors:
            _predictors[name] = Predictor(name)
        return _predictors[name]


def loaded_models() -> List[str]:
    return list(_predictors)
