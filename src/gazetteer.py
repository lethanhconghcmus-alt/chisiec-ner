"""
gazetteer.py — Load TITLE/ORG/LOC gazetteer jsonl files, tag token sequences
với nhãn BIO-gazetteer (feature phụ cho model, KHÔNG dùng làm weak label).
"""

import json
import os

from src.utils import get_logger

logger = get_logger(__name__)

GAZ_TYPES = ["TITLE", "ORG", "LOC"]
# Ưu tiên khi 2 loại cùng khớp 1 span cùng độ dài (hiếm, nhưng cần quyết định)
TYPE_PRIORITY = {"TITLE": 0, "ORG": 1, "LOC": 2}

GAZ_LABELS = ["O"] + [f"{p}-{t}" for t in GAZ_TYPES for p in ("B", "I")]
GAZ_LABEL2ID = {l: i for i, l in enumerate(GAZ_LABELS)}


def load_gazetteer(gaz_dir: str) -> dict:
    """Đọc {title,org,loc}.jsonl trong gaz_dir. Trả về {TYPE: set(surface)}."""
    surfaces = {t: set() for t in GAZ_TYPES}
    for t in GAZ_TYPES:
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
                surfaces[t].add(rec["surface"])
        logger.info(f"Gazetteer[{t}]: {len(surfaces[t])} surfaces loaded")
    return surfaces


class GazetteerTagger:
    """Longest-match-first tagging của 1 câu (list ký tự) -> list nhãn B-/I-/O."""

    def __init__(self, surfaces: dict):
        self.surfaces = surfaces
        self.max_len = {
            t: max((len(s) for s in surfaces[t]), default=0) for t in GAZ_TYPES
        }

    def tag(self, tokens: list) -> list:
        n = len(tokens)
        tags = ["O"] * n
        i = 0
        while i < n:
            best_type, best_len = None, 0
            for t in GAZ_TYPES:
                cap = min(self.max_len[t], n - i)
                for length in range(cap, 0, -1):
                    span = "".join(tokens[i:i + length])
                    if span in self.surfaces[t]:
                        if length > best_len or (
                            length == best_len
                            and best_type is not None
                            and TYPE_PRIORITY[t] < TYPE_PRIORITY[best_type]
                        ):
                            best_type, best_len = t, length
                        break  # đã tìm match dài nhất cho type này
            if best_type is None:
                i += 1
                continue
            tags[i] = f"B-{best_type}"
            for j in range(1, best_len):
                tags[i + j] = f"I-{best_type}"
            i += best_len
        return tags
