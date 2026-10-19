"""
evaluate.py — chấm pipeline (ngắt câu → NER) trên true label gõ tay (web/gold_manual.txt).

Gold là văn bản thô không dấu câu. Pipeline tự ngắt câu lại, nên thực thể dự đoán
được quy về toạ độ trên chuỗi thô (bỏ các dấu câu model chèn) rồi mới so khớp.

  strict : khớp cả biên lẫn nhãn
  span   : chỉ khớp biên — dùng để so với guwen-ner (không có PER/LOC/…)
  unseen : như strict nhưng chỉ tính thực thể (gold và dự đoán) nằm trọn trong vùng chữ
           KHÔNG xuất hiện trong tập train DVSKTT (DVSK_TRAIN). Gold gõ tay trùng phần lớn
           với train DVSKTT, nên model train trên DVSKTT phải xem cột này mới công bằng.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from web.pipeline import PUNCT_RE, Pipeline

GOLD_MANUAL = Path(__file__).parent / "gold_manual.txt"
GOLD_RE = re.compile(r"\{([^{}|]+)\|([A-Z]+)\}")
DVSK_TRAIN = os.path.expanduser(os.getenv(
    "DVSK_TRAIN", "~/Downloads/kaggle_ckpts/gujiroberta_crf_maxlen256/dvsk_train.txt"))
SEEN_NGRAM = 12  # chữ nằm trong một chuỗi 12 chữ có trong train → coi là "đã thấy"


@lru_cache(maxsize=1)
def _train_text() -> str:
    if not os.path.exists(DVSK_TRAIN):
        return ""
    with open(DVSK_TRAIN, encoding="utf-8") as f:
        return "".join(l.split()[0] if l.strip() else "\n" for l in f)


def seen_mask(text: str) -> list:
    """True tại các chữ thuộc một chuỗi SEEN_NGRAM chữ xuất hiện trong train DVSKTT."""
    train, mask = _train_text(), [False] * len(text)
    if not train:
        return mask
    for i in range(len(text) - SEEN_NGRAM + 1):
        if text[i:i + SEEN_NGRAM] in train:
            mask[i:i + SEEN_NGRAM] = [True] * SEEN_NGRAM
    return mask


def load_gold() -> List[Dict]:
    """Mỗi dòng 1 đoạn, thực thể dạng {chữ|NHÃN}; đọc lại mỗi lần gọi để sửa nhãn không cần restart."""
    docs: List[Dict] = []
    if not GOLD_MANUAL.exists():
        return docs
    for line in GOLD_MANUAL.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"\s", "", line)
        if not line or line.startswith("#"):
            continue
        text, ents, pos = "", [], 0
        for m in GOLD_RE.finditer(line):
            text += line[pos:m.start()]
            ents.append({"text": m.group(1), "label": m.group(2),
                         "start": len(text), "end": len(text) + len(m.group(1))})
            text += m.group(1)
            pos = m.end()
        text += line[pos:]
        docs.append({"id": len(docs) + 1, "text": text, "entities": ents})
    return docs


def _strip_punct(text: str, ents: List[Dict]) -> tuple[str, List[Dict]]:
    """Bỏ dấu câu trong gold (nếu có) và dời toạ độ thực thể theo."""
    raw_idx, raw = [], []
    for ch in text:
        raw_idx.append(len(raw))
        if not PUNCT_RE.match(ch):
            raw.append(ch)
    raw_idx.append(len(raw))
    out = [{**e, "start": raw_idx[e["start"]], "end": raw_idx[e["end"]]} for e in ents]
    return "".join(raw), [e for e in out if e["end"] > e["start"]]


def to_raw(sentences: List[str], ents_per_sent: List[List[Dict]]) -> List[Dict]:
    """Toạ độ theo câu (có dấu model chèn) → toạ độ trên chuỗi thô."""
    out, raw = [], 0
    for s, ents in zip(sentences, ents_per_sent):
        idx = []
        for ch in s:
            idx.append(-1 if PUNCT_RE.match(ch) else raw)
            if not PUNCT_RE.match(ch):
                raw += 1
        for e in ents:
            pos = [i for i in idx[e["start"]:e["end"]] if i >= 0]
            if pos:
                out.append({"label": e["label"], "start": pos[0], "end": pos[-1] + 1})
    return out


def _prf(tp: int, n_pred: int, n_gold: int) -> Dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "tp": tp, "pred": n_pred, "gold": n_gold}


def evaluate(pipeline: Pipeline, segmenter: str, models: List[str], simplify: bool = False) -> Dict:
    docs = load_gold()
    tp_strict, tp_span = Counter(), Counter()
    n_pred, n_gold = Counter(), 0
    tp_unseen, pred_unseen, gold_unseen = Counter(), Counter(), 0
    by_type = {m: {"tp_strict": Counter(), "pred": Counter(), "tp_span": Counter()} for m in models}
    gold_by_type: Counter = Counter()

    for d in docs:
        text, gold = _strip_punct(d["text"], d["entities"])
        sentences = pipeline.segment(text, segmenter)
        n_gold += len(gold)
        gold_by_type.update(e["label"] for e in gold)
        mask = seen_mask(text)
        unseen = lambda s, e: not any(mask[s:e])
        g_strict = {(e["start"], e["end"], e["label"]) for e in gold}
        g_unseen = {x for x in g_strict if unseen(x[0], x[1])}
        gold_unseen += len(g_unseen)
        g_span = {(e["start"], e["end"]) for e in gold}
        for m in models:
            pred = to_raw(sentences, pipeline.tagger(m).predict(sentences, simplify)) if sentences else []
            p_strict = {(e["start"], e["end"], e["label"]) for e in pred}
            p_span = {(e["start"], e["end"]) for e in pred}
            n_pred[m] += len(p_strict)
            p_unseen = {x for x in p_strict if unseen(x[0], x[1])}
            pred_unseen[m] += len(p_unseen)
            tp_unseen[m] += len(p_unseen & g_unseen)
            tp_strict[m] += len(p_strict & g_strict)
            tp_span[m] += len(p_span & g_span)
            bt = by_type[m]
            bt["pred"].update(lab for _, _, lab in p_strict)
            bt["tp_strict"].update(lab for _, _, lab in p_strict & g_strict)
            # recall theo loại gold, chỉ cần khớp biên
            bt["tp_span"].update(e["label"] for e in gold if (e["start"], e["end"]) in p_span)

    out = {"segmenter": segmenter, "simplify": simplify, "docs": len(docs), "gold_entities": n_gold,
           "gold_unseen": gold_unseen, "has_train": bool(_train_text()),
           "gold_by_type": dict(gold_by_type), "models": {}}
    for m in models:
        bt = by_type[m]
        out["models"][m] = {
            "strict": _prf(tp_strict[m], n_pred[m], n_gold),
            "span": _prf(tp_span[m], n_pred[m], n_gold),
            "unseen": _prf(tp_unseen[m], pred_unseen[m], gold_unseen),
            "per_type": {t: {**_prf(bt["tp_strict"][t], bt["pred"][t], g),
                             "span_recall": bt["tp_span"][t] / g if g else 0.0}
                         for t, g in gold_by_type.items()},
        }
    return out
