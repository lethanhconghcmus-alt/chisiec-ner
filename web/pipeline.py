"""
pipeline.py — Tách câu (断句) + NER cho Hán văn cổ, nhiều model để so sánh.

  Ngắt câu (chọn 1):
    guwen-punc  ethanyt/guwen-punc — khôi phục dấu câu, train trên chữ giản thể
    biaodian    raynardj/classical-chinese-punctuation-guwen-biaodian
    none        giữ nguyên dấu câu của input
  NER (chọn nhiều): các checkpoint GuwenBERT fine-tune (xem NER_MODELS) + ethanyt/guwen-ner
    công khai (chỉ NOUN / BOOK, giải mã argmax vì HF không kèm trọng số CRF).

GuwenBERT có vocab giản thể: ~24% chữ DVSKTT phồn thể thành [UNK]. Tuỳ chọn `simplify`
đổi phồn → giản thể theo TỪNG CHỮ (giữ nguyên độ dài để offset không lệch) trước khi
đưa vào NER; kết quả hiển thị trên chữ gốc. guwen-punc / guwen-ner luôn đổi.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models import GuwenBertCRF, GuwenBertLinear  # noqa: E402

NER_BACKBONE = os.getenv("BACKBONE", "ethanyt/guwenbert-base")
NER_LABEL_MAP = os.getenv("LABEL_MAP", str(ROOT / "artifacts/label_map.json"))
NER_MAX_LEN = int(os.getenv("MAX_LEN", "128"))
PUNC_WINDOW = 400  # số chữ mỗi lần đưa vào model ngắt câu (< 512 token)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")

SENT_END = "。？！；?!;"
KEEP_PUNCT = set("，。：；？！、")  # bỏ ngoặc/nháy model sinh ra (hay lệch cặp)
# Mọi dấu câu (Trung + Latin) — bị bỏ đi trước khi model tự ngắt câu lại
PUNCT_RE = re.compile(r"[\s，。、；：？！“”‘’「」『』（）《》〈〉【】…—·,.;:?!\"'()\[\]]")

SEGMENTERS = {
    "guwen-punc": {"name": "guwen-punc", "hf": "ethanyt/guwen-punc", "simplify": True,
                   "desc": "ethanyt/guwen-punc (khôi phục dấu câu, giản thể)"},
    "biaodian": {"name": "biaodian", "hf": "raynardj/classical-chinese-punctuation-guwen-biaodian",
                 "simplify": False, "desc": "raynardj/…-guwen-biaodian"},
    "none": {"name": "Giữ nguyên", "desc": "Không ngắt lại — dùng dấu câu có sẵn trong input"},
}
CKPT = ROOT / "checkpoints"
# TQ = 4 bộ sử Trung Quốc gộp (AnChineseNERE + CHisIEC + BEEVETA + C-CLUE, 11.386 câu train).
# Tất cả fine-tune toàn bộ GuwenBERT (ethanyt/guwenbert-base), nhãn BIO PER/LOC/ORG/DTM/TITLE.
# Nhóm DVSKTT: cùng xuất phát từ một checkpoint trung gian (không có trên máy) và train trên
# data/dvsk/*.txt — bộ này nhãn lệch vị trí nặng và test trùng 81% với train.
NER_MODELS = {
    "gw-crf-tq": {"name": "GuwenBERT+CRF · TQ", "ckpt": CKPT / "best.pt", "head": "crf",
                  "desc": "Fine-tune trên 4 bộ sử TQ, test F1 0.932 trên TQ (ancient_ner_models, 04/2026) — model của API"},
    "gw-linear-tq": {"name": "GuwenBERT+Linear · TQ", "ckpt": CKPT / "compare/gw-linear-tq.pt", "head": "linear",
                     "desc": "Như trên nhưng đầu Linear (argmax), không CRF (ancient_ner_models, 04/2026)"},
    "gw-crf-dvsk": {"name": "GuwenBERT+CRF · DVSKTT", "ckpt": CKPT / "compare/gw-crf-dvsk.pt", "head": "crf",
                    "desc": "Fine-tune trên DVSKTT (results/outputs/dvsk/guwenbert_crf, 06/2026)"},
    "gw-linear-dvsk": {"name": "GuwenBERT+Linear · DVSKTT", "ckpt": CKPT / "compare/gw-linear-dvsk.pt", "head": "linear",
                       "desc": "Fine-tune trên DVSKTT, đầu Linear (results/outputs/dvsk/guwenbert_finetune, 06/2026)"},
    "gw-crf-dvsk-2": {"name": "GuwenBERT+CRF · DVSKTT (bản 2)", "ckpt": CKPT / "compare/gw-crf-jun.pt", "head": "crf",
                      "desc": "results/outputs/guwenbert_crf (06/2026) — cùng gốc trọng số và cùng hành vi với nhóm DVSKTT"},
    "guwen-ner": {"name": "guwen-ner (công khai)", "hf": "ethanyt/guwen-ner",
                  "desc": "ethanyt/guwen-ner — chỉ NOUN (tên riêng) / BOOK, argmax (HF không kèm CRF)"},
}
# bỏ checkpoint không có trên máy này
NER_MODELS = {k: v for k, v in NER_MODELS.items() if "hf" in v or Path(v["ckpt"]).exists()}
# nhãn của guwen-ner → nhãn hiển thị
GUWEN_NER_LABELS = {"NOUN_OTHER": "NOUN", "NOUN_BOOKNAME": "BOOK"}

_t2s = None


def to_simplified(text: str) -> str:
    """Đổi phồn → giản thể từng chữ, giữ nguyên độ dài chuỗi."""
    global _t2s
    if _t2s is None:
        import opencc
        _t2s = opencc.OpenCC("t2s")
    out = []
    for ch in text:
        s = _t2s.convert(ch)
        out.append(s if len(s) == 1 else ch)
    return "".join(out)


class Segmenter:
    def __init__(self, hf_name: str, simplify: bool):
        self.tok = AutoTokenizer.from_pretrained(hf_name)
        self.model = AutoModelForTokenClassification.from_pretrained(hf_name).to(DEVICE).eval()
        self.id2label = self.model.config.id2label
        self.simplify = simplify

    @staticmethod
    def _mark(label: str) -> str:
        # guwen-punc dùng dấu Latin (B-, B-. B-\ …) → đổi sang dấu Hán
        lab = label.replace("B-", "")
        return {",": "，", ".": "。", "?": "？", "!": "！", "\\": "、", ":": "：", ";": "；"}.get(lab, lab)

    @torch.no_grad()
    def punctuate(self, text: str) -> str:
        """Trả về text đã chèn dấu câu (dấu bỏ trống nếu label là O)."""
        chars = list(text)
        model_chars = list(to_simplified(text)) if self.simplify else chars
        out: List[str] = []
        for start in range(0, len(chars), PUNC_WINDOW):
            part = model_chars[start:start + PUNC_WINDOW]
            enc = self.tok(part, is_split_into_words=True, return_tensors="pt",
                           truncation=True, max_length=512)
            pred = self.model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits.argmax(-1)[0].tolist()
            label_of: Dict[int, str] = {}
            for i, w in enumerate(enc.word_ids()):
                if w is not None and w not in label_of:
                    label_of[w] = self.id2label[pred[i]]
            for i, ch in enumerate(chars[start:start + PUNC_WINDOW]):
                out.append(ch)
                mark = self._mark(label_of.get(i, "O"))
                if mark in KEEP_PUNCT:
                    out.append(mark)
        # đảm bảo chunk cuối kết câu
        if out and out[-1] not in SENT_END:
            out.append("。")
        return "".join(out)


def split_sentences(text: str) -> List[str]:
    sents, buf = [], ""
    for ch in text:
        buf += ch
        if ch in SENT_END:
            if buf.strip():
                sents.append(buf.strip())
            buf = ""
    if buf.strip():
        sents.append(buf.strip())
    return sents


class NERTagger:
    """Chạy NER trên danh sách câu liền nhau; `tag_fn` gán nhãn BIO cho một batch đoạn văn."""

    def __init__(self, tag_fn: Callable[[List[str]], List[List[str]]], always_simplify: bool = False):
        self.tag_fn = tag_fn
        self.always_simplify = always_simplify
        self.max_chars = NER_MAX_LEN - 2

    def _pieces(self, sent: str) -> List[Tuple[int, str]]:
        """Cắt text thành các đoạn <= max_chars, ưu tiên cắt sau dấu kết câu, rồi dấu phẩy."""
        if len(sent) <= self.max_chars:
            return [(0, sent)]
        pieces, start = [], 0
        while start < len(sent):
            end = min(start + self.max_chars, len(sent))
            if end < len(sent):
                cut = max(sent.rfind(p, start, end) for p in SENT_END)
                if cut <= start:
                    cut = max(sent.rfind(p, start, end) for p in "，、：,:")
                if cut > start:
                    end = cut + 1
            pieces.append((start, sent[start:end]))
            start = end
        return pieces

    def predict(self, sentences: List[str], simplify: bool = False, batch_size: int = 16) -> List[List[Dict]]:
        """Ghép các câu liền nhau thành cửa sổ <= max_chars để NER có ngữ cảnh
        (model ngắt câu hay chèn 。 nhầm giữa các tên liệt kê), rồi tách span về từng câu."""
        full = "".join(sentences)
        model_text = to_simplified(full) if simplify or self.always_simplify else full
        jobs = self._pieces(model_text)
        labels = ["O"] * len(full)
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            for (off, txt), labs in zip(batch, self.tag_fn([t for _, t in batch])):
                labels[off:off + len(txt)] = labs
        result, pos = [], 0
        for s in sentences:
            result.append(bio_to_spans(s, labels[pos:pos + len(s)]))
            pos += len(s)
        return result


def _first_subtoken_labels(enc, texts: List[str], pred: List[List[int]], id2label) -> List[List[str]]:
    result = []
    for b, t in enumerate(texts):
        labels = ["O"] * len(t)
        seen = set()
        for i, w in enumerate(enc.word_ids(batch_index=b)):
            if w is not None and w not in seen and i < len(pred[b]):
                seen.add(w)
                labels[w] = id2label[pred[b][i]]
        result.append(labels)
    return result


def load_checkpoint(ckpt: Path, head: str) -> NERTagger:
    """Checkpoint GuwenBertCRF / GuwenBertLinear của src/models.py (cùng label map 11 nhãn)."""
    with open(NER_LABEL_MAP, encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
    tok = AutoTokenizer.from_pretrained(NER_BACKBONE)
    cls = GuwenBertCRF if head == "crf" else GuwenBertLinear
    model = cls(NER_BACKBONE, len(id2label))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(DEVICE).eval()

    @torch.no_grad()
    def tag(texts: List[str]) -> List[List[str]]:
        enc = tok([list(t) for t in texts], is_split_into_words=True, padding=True,
                  truncation=True, max_length=NER_MAX_LEN, return_tensors="pt")
        tti = enc.get("token_type_ids")
        pred = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE),
                     tti.to(DEVICE) if tti is not None else None)
        return _first_subtoken_labels(enc, texts, pred, id2label)

    return NERTagger(tag)


def load_guwen_ner(hf_name: str) -> NERTagger:
    # guwen-ner train trên giản thể nên luôn đổi
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForTokenClassification.from_pretrained(hf_name).to(DEVICE).eval()
    id2label = {}
    for i, lab in model.config.id2label.items():
        if lab == "O":
            id2label[int(i)] = lab
        else:
            prefix, typ = lab.split("-", 1)
            id2label[int(i)] = f"{prefix}-{GUWEN_NER_LABELS.get(typ, typ)}"

    @torch.no_grad()
    def tag(texts: List[str]) -> List[List[str]]:
        enc = tok([list(t) for t in texts], is_split_into_words=True, padding=True,
                  truncation=True, max_length=NER_MAX_LEN, return_tensors="pt")
        logits = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
        return _first_subtoken_labels(enc, texts, logits.argmax(-1).tolist(), id2label)

    return NERTagger(tag, always_simplify=True)


def bio_to_spans(text: str, labels: List[str]) -> List[Dict]:
    spans, cur = [], None
    for i, lab in enumerate(labels):
        typ = lab[2:] if lab != "O" else None
        if lab.startswith("B-") or (lab.startswith("I-") and (cur is None or cur["label"] != typ)):
            if cur:
                spans.append(cur)
            cur = {"label": typ, "start": i, "end": i + 1}
        elif lab.startswith("I-"):
            cur["end"] = i + 1
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]
    return spans


class Pipeline:
    """Nạp model theo yêu cầu (lần đầu dùng) và giữ lại trong bộ nhớ."""

    def __init__(self):
        self._segmenters: Dict[str, Segmenter] = {}
        self._taggers: Dict[str, NERTagger] = {}
        self._lock = threading.Lock()

    def segmenter(self, name: str) -> Segmenter:
        with self._lock:
            if name not in self._segmenters:
                cfg = SEGMENTERS[name]
                self._segmenters[name] = Segmenter(cfg["hf"], cfg["simplify"])
            return self._segmenters[name]

    def tagger(self, name: str) -> NERTagger:
        cfg = NER_MODELS[name]
        with self._lock:
            if name not in self._taggers:
                if "hf" in cfg:
                    self._taggers[name] = load_guwen_ner(cfg["hf"])
                else:
                    self._taggers[name] = load_checkpoint(cfg["ckpt"], cfg["head"])
            return self._taggers[name]

    def segment(self, text: str, segmenter: str) -> List[str]:
        if segmenter == "none":
            # giữ nguyên: mỗi dòng / mỗi dấu kết câu là một câu
            return [s for ln in text.splitlines() for s in split_sentences(re.sub(r"\s", "", ln))]
        # bỏ dấu câu/khoảng trắng có sẵn theo từng dòng rồi để model ngắt lại
        seg = self.segmenter(segmenter)
        lines = [PUNCT_RE.sub("", ln) for ln in text.splitlines()]
        return split_sentences("".join(seg.punctuate(ln) for ln in lines if ln))

    def run(self, text: str, segmenter: str, models: List[str], simplify: bool = False) -> Dict:
        sentences = self.segment(text, segmenter)
        results = {m: self.tagger(m).predict(sentences, simplify) if sentences else [] for m in models}
        return {"segmenter": segmenter, "simplify": simplify, "punctuated": "".join(sentences),
                "sentences": sentences, "results": results}
