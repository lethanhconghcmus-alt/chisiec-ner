"""
Segmenter — sentence segmentation (断句) for Ancient Chinese.

A token-classification model predicts the punctuation mark that follows
each character; the punctuated text is then split on sentence-final marks.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

logger = logging.getLogger("api.segmenter")

PUNC_MODEL  = os.getenv("PUNC_MODEL", "raynardj/classical-chinese-punctuation-guwen-biaodian")
PUNC_WINDOW = int(os.getenv("PUNC_WINDOW", "400"))  # chars per forward pass (< 512 tokens)
DEVICE      = torch.device(os.getenv("DEVICE", "cpu"))

SENT_END   = "。？！；?!;"
# Keep only reliable marks — quotes/brackets from the model are often unbalanced
KEEP_PUNCT = set("，。：；？！、")
# All existing punctuation (CJK + Latin) and whitespace, stripped before re-punctuating
PUNCT_RE   = re.compile(r"[\s，。、；：？！“”‘’「」『』（）《》〈〉【】…—·,.;:?!\"'()\[\]]")


def split_sentences(text: str) -> List[str]:
    """Split already-punctuated text after every sentence-final mark."""
    sentences: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


class Segmenter:
    def __init__(self) -> None:
        logger.info("Loading punctuation model from %s", PUNC_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(PUNC_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(PUNC_MODEL)
        self.model.to(DEVICE)
        self.model.eval()
        self.id2label: Dict[int, str] = self.model.config.id2label
        logger.info("Segmenter ready on %s", DEVICE)

    @torch.no_grad()
    def punctuate(self, text: str) -> str:
        """Return `text` with predicted punctuation inserted after each character."""
        chars = list(text)
        out: List[str] = []
        for i in range(0, len(chars), PUNC_WINDOW):
            part = chars[i:i + PUNC_WINDOW]
            encoding = self.tokenizer(
                part,
                is_split_into_words=True,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            logits = self.model(**{k: v.to(DEVICE) for k, v in encoding.items()}).logits
            pred_ids = logits.argmax(-1)[0].tolist()

            label_of: Dict[int, str] = {}
            for tok_idx, wid in enumerate(encoding.word_ids(batch_index=0)):
                if wid is not None and wid not in label_of:
                    label_of[wid] = self.id2label[pred_ids[tok_idx]]

            for wid, ch in enumerate(part):
                out.append(ch)
                mark = label_of.get(wid, "O").replace("B-", "")
                if mark in KEEP_PUNCT:
                    out.append(mark)

        # Make sure the last sentence is closed
        if out and out[-1] not in SENT_END:
            out.append("。")
        return "".join(out)

    def segment(self, text: str) -> List[str]:
        """
        Strip existing punctuation line by line, re-punctuate with the model,
        and return the list of sentences.
        """
        lines = [PUNCT_RE.sub("", ln) for ln in text.splitlines()]
        punctuated = "".join(self.punctuate(ln) for ln in lines if ln)
        return split_sentences(punctuated)
