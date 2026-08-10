"""
gazetteer.py — Tag token sequences với nhãn BIO-gazetteer/rule, dùng làm
feature phụ cho model (KHÔNG dùng làm weak label).

3 nguồn tín hiệu, độ tin cậy khác hẳn nhau (xem docs/gazetteer_findings.md):
  - TITLE/ORG/LOC: surface-list lookup (longest-match). Entry 1 ký tự bị BỎ
    khi build matcher — đo thực nghiệm trên test set cho thấy match 1 ký tự
    chỉ đúng loại 6.1% (93.9% nhiễu), trong khi match >=2 ký tự đúng 46-80%.
  - PER: KHÔNG dùng surface-list (list nhớ "鄭松 là PER" thay vì học pattern,
    sập khi gặp tên mới — xem gazetteer_findings.md §"Ghi nhớ thay vì học
    mẫu"). Dùng RULE: ký tự đầu là 1 trong ~30 họ phổ biến.
  - DTM: KHÔNG dùng gazetteer, dùng RULE: pattern tháng + 60 can-chi (rule
    module riêng, precision đo được ~92-95%, xem gazetteer_findings.md §7-8).
"""

import json
import os
import re

from src.utils import get_logger

logger = get_logger(__name__)

GAZ_TYPES = ["TITLE", "ORG", "LOC", "PER", "DTM"]
# Ưu tiên khi 2 loại cùng khớp 1 vị trí: rule (PER/DTM, precision cao) > lexicon
TYPE_PRIORITY = {"PER": 0, "DTM": 1, "TITLE": 2, "ORG": 3, "LOC": 4}
MIN_LEXICON_LEN = 2  # bỏ surface 1 ký tự khỏi TITLE/ORG/LOC matcher

GAZ_LABELS = ["O"] + [f"{p}-{t}" for t in GAZ_TYPES for p in ("B", "I")]
GAZ_LABEL2ID = {l: i for i, l in enumerate(GAZ_LABELS)}

# ── PER: luật họ (curated, xem scripts/build_gazetteer.py SURNAME_CHARS) ──────
SURNAME_CHARS = set(
    "阮鄭黎陳莫吳范武杜李楊裴陶鄧潘丁黄郭張梁何胡蘇謝韓馬高林秦金"
)

# ── DTM: luật can-chi (60 hoa giáp) + tên tháng ────────────────────────────────
_THIEN_CAN = "甲乙丙丁戊己庚辛壬癸"
_DIA_CHI = "子丑寅卯辰巳午未申酉戌亥"
CAN_CHI_60 = {_THIEN_CAN[i % 10] + _DIA_CHI[i % 12] for i in range(60)}

_MONTH_NUM = "正二三四五六七八九十"
_MONTH_PATTERNS = [
    r"(春|夏|秋|冬)?閏?(正|二|三|四|五|六|七|八|九|十|十一|十二)月",
]
DTM_RE = re.compile("|".join(_MONTH_PATTERNS))


def load_gazetteer(gaz_dir: str) -> dict:
    """Đọc {title,org,loc}.jsonl trong gaz_dir. Trả về {TYPE: set(surface)},
    đã lọc surface < MIN_LEXICON_LEN ký tự."""
    surfaces = {t: set() for t in ("TITLE", "ORG", "LOC")}
    for t in ("TITLE", "ORG", "LOC"):
        path = os.path.join(gaz_dir, f"{t.lower()}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"Gazetteer file not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                surface = rec["surface"]
                if len(surface) < MIN_LEXICON_LEN:
                    continue
                surfaces[t].add(surface)
        logger.info(f"Gazetteer[{t}]: {len(surfaces[t])} surfaces loaded (min_len={MIN_LEXICON_LEN})")
    return surfaces


class GazetteerTagger:
    """Kết hợp lexicon (TITLE/ORG/LOC) + rule (PER/DTM) -> 1 nhãn BIO / token."""

    def __init__(self, surfaces: dict):
        self.surfaces = surfaces
        self.max_len = {
            t: max((len(s) for s in surfaces.get(t, ())), default=0) for t in ("TITLE", "ORG", "LOC")
        }

    # ── RULE: DTM (can-chi + tên tháng) ───────────────────────────────────────
    def _dtm_spans(self, text: str):
        spans = []
        for m in DTM_RE.finditer(text):
            spans.append((m.start(), m.end()))
        i = 0
        n = len(text)
        while i < n - 1:
            if text[i:i + 2] in CAN_CHI_60:
                spans.append((i, i + 2))
                i += 2
            else:
                i += 1
        return spans

    # ── RULE: PER (họ mở đầu) — chỉ tag 1 ký tự họ, không đoán độ dài tên ─────
    def _per_spans(self, text: str):
        return [(i, i + 1) for i, ch in enumerate(text) if ch in SURNAME_CHARS]

    def tag(self, tokens: list) -> list:
        n = len(tokens)
        text = "".join(tokens)
        tags = ["O"] * n

        # rule-based trước (ưu tiên cao hơn), rồi lexicon lấp phần còn trống
        rule_spans = []
        for s, e in self._dtm_spans(text):
            rule_spans.append((s, e, "DTM"))
        for s, e in self._per_spans(text):
            rule_spans.append((s, e, "PER"))
        # span dài hơn thắng khi chồng lấn (DTM 2 ký tự > PER 1 ký tự)
        rule_spans.sort(key=lambda x: -(x[1] - x[0]))
        taken = [False] * n
        for s, e, typ in rule_spans:
            if any(taken[s:e]):
                continue
            tags[s] = f"B-{typ}"
            for j in range(s + 1, e):
                tags[j] = f"I-{typ}"
            for j in range(s, e):
                taken[j] = True

        i = 0
        while i < n:
            if taken[i]:
                i += 1
                continue
            best_type, best_len = None, 0
            for t in ("TITLE", "ORG", "LOC"):
                cap = min(self.max_len[t], n - i)
                for length in range(cap, 0, -1):
                    if any(taken[i:i + length]):
                        continue
                    span = text[i:i + length] if False else "".join(tokens[i:i + length])
                    if span in self.surfaces[t]:
                        if length > best_len:
                            best_type, best_len = t, length
                        break
            if best_type is None:
                i += 1
                continue
            tags[i] = f"B-{best_type}"
            for j in range(1, best_len):
                tags[i + j] = f"I-{best_type}"
            i += best_len
        return tags
