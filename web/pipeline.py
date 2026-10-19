"""
pipeline.py — Tách câu (断句) + NER cho Hán văn cổ, nhiều model để so sánh.

  Ngắt câu (chọn 1):
    guwen-punc  ethanyt/guwen-punc — khôi phục dấu câu, train trên chữ giản thể
    biaodian    raynardj/classical-chinese-punctuation-guwen-biaodian
    none        giữ nguyên dấu câu của input
  NER (chọn nhiều): các model encoder + CRF của API (api/predictor.py — dùng chung
    checkpoint và cách nạp) + ethanyt/guwen-ner công khai (chỉ NOUN / BOOK, giải mã
    argmax vì HF không kèm trọng số CRF).

GuwenBERT / GujiRoBERTa_jian có vocab giản thể nên nhiều chữ DVSKTT phồn thể thành [UNK].
Tuỳ chọn `simplify` đổi phồn → giản thể theo TỪNG CHỮ (giữ nguyên độ dài để offset không
lệch) trước khi đưa vào NER; kết quả hiển thị trên chữ gốc. guwen-punc / guwen-ner luôn đổi.
"""
from __future__ import annotations

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
os.chdir(ROOT)  # đường dẫn checkpoint/label map của API là tương đối với gốc repo

PUNC_WINDOW = 400  # số chữ mỗi lần đưa vào model ngắt câu (< 512 token)
GUWEN_NER_MAX_LEN = 128

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("DEVICE", str(DEVICE))  # API predictor đọc DEVICE lúc import
from api.predictor import NER_MODELS as API_MODELS, available_models, get_predictor  # noqa: E402

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
# Tên hiển thị cho các model của API (api/predictor.py); model nào chưa có checkpoint thì ẩn.
API_MODEL_NAMES = {
    "guwenbert-crf-tq": "GuwenBERT+CRF · TQ",
    "sikubert-crf-dvsktt": "SikuBERT+CRF · DVSKTT",
    "gujiroberta-crf-dvsktt": "GujiRoBERTa+CRF · DVSKTT",
}
NER_MODELS = {name: {"name": API_MODEL_NAMES.get(name, name), "desc": API_MODELS[name]["description"]}
              for name in available_models()}
NER_MODELS["guwen-ner"] = {"name": "guwen-ner (công khai)", "hf": "ethanyt/guwen-ner",
                           "desc": "ethanyt/guwen-ner — chỉ NOUN (tên riêng) / BOOK, argmax (HF không kèm CRF)"}
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

    def __init__(self, tag_fn: Callable[[List[str]], List[List[str]]], max_len: int,
                 always_simplify: bool = False):
        self.tag_fn = tag_fn
        self.always_simplify = always_simplify
        self.max_chars = max_len - 2

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


def load_api_model(name: str) -> NERTagger:
    """Model encoder + CRF của API (api/predictor.py): dùng chung Predictor.tag() theo batch."""
    p = get_predictor(name)
    return NERTagger(p.tag, p.max_len)


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
                  truncation=True, max_length=GUWEN_NER_MAX_LEN, return_tensors="pt")
        logits = model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
        return _first_subtoken_labels(enc, texts, logits.argmax(-1).tolist(), id2label)

    return NERTagger(tag, GUWEN_NER_MAX_LEN, always_simplify=True)


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
                    self._taggers[name] = load_api_model(name)
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
