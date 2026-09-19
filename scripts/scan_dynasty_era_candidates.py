"""
scan_dynasty_era_candidates.py — GR-11 candidate scanner (REVIEW ONLY).
Tìm cụm [polity/dynasty prefix][era name][numeral]+年 để đề xuất tách span,
KHÔNG tự động sửa gì. Không giả định mọi prefix là ORG -- chỉ báo cáo theo
đúng chữ xuất hiện, để reviewer tự quyết.

Usage:
  python scripts/scan_dynasty_era_candidates.py \
      --train ... --dev ... --test ... --out-dir artifacts/audit_v1
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_utils import read_conll
from src.audit_utils import extract_all_spans, load_source_map

# Lexicon polity/dynasty tối thiểu, curated tu vi du GR-04/GR-11 va tu cac
# surface da audit (莫,明,陳,黎,李,晋,吳,清,宋,元,唐,大越,大明,大清). CO THE
# THIEU -- day la danh sach khoi dau, khong day du, dung de sinh candidate
# CHU KHONG phai ground truth.
POLITY_LEXICON = [
    "大越", "大明", "大清", "大元", "大宋", "大唐",
    "明", "莫", "陳", "黎", "李", "晋", "吳", "清", "宋", "元", "唐",
]
NUMERAL_CHARS = "0123456789〇一二三四五六七八九十百千"

_polity_re = "|".join(sorted(POLITY_LEXICON, key=len, reverse=True))
CANDIDATE_RE = re.compile(
    rf"(?P<polity>{_polity_re})(?P<era>[一-鿿]{{2,4}}?)(?P<numeral>[{NUMERAL_CHARS}]+)年"
)


def find_candidates_in_text(text: str) -> list:
    out = []
    for m in CANDIDATE_RE.finditer(text):
        out.append({
            "polity": m.group("polity"),
            "era": m.group("era"),
            "numeral": m.group("numeral"),
            "start": m.start(),
            "end": m.end() - 1,  # inclusive
            "full_match": m.group(0),
        })
    return out


def confidence_for_candidate(cand: dict, existing_entities_here: list) -> str:
    """
    high: full_match trùng khít 1 gold entity ĐANG CÓ (đây chính xác là
      pattern 明成化/莫大正 đã audit).
    medium: polity/era nằm chồng lấn 1 phần gold entity (gold có thể chỉ
      tag 1 phần của cụm).
    low: không chồng lấn gold entity nào (raw text match, chưa qua gold).
    """
    for e in existing_entities_here:
        if e["start"] == cand["start"] and e["end"] == cand["end"]:
            return "high"
    for e in existing_entities_here:
        if not (e["end"] < cand["start"] or e["start"] > cand["end"]):
            return "medium"
    return "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None)
    ap.add_argument("--out-dir", default="artifacts/audit_v1")
    args = ap.parse_args()

    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=20)

    from collections import defaultdict
    entities_by_sentence = defaultdict(list)
    for e in entities:
        entities_by_sentence[(e["split"], e["sample_id"])].append(e)

    rows = []
    for split, sentences in splits_data.items():
        for idx, (tokens, labels) in enumerate(sentences):
            text = "".join(tokens)
            candidates = find_candidates_in_text(text)
            if not candidates:
                continue
            ents_here = entities_by_sentence.get((split, idx), [])
            for cand in candidates:
                conf = confidence_for_candidate(cand, ents_here)
                overlapping_gold = [
                    {"surface": e["surface"], "label": e["label"], "start": e["start"], "end": e["end"]}
                    for e in ents_here
                    if not (e["end"] < cand["start"] or e["start"] > cand["end"])
                ]
                ctx_s, ctx_e = max(0, cand["start"] - 20), min(len(text), cand["end"] + 1 + 20)
                marked = f"{text[ctx_s:cand['start']]}【{cand['full_match']}】{text[cand['end']+1:ctx_e]}"
                polity_end = cand["start"] + len(cand["polity"]) - 1
                era_start = polity_end + 1
                rows.append({
                    "split": split,
                    "sample_id": idx,
                    "text": text,
                    "marked_context": marked,
                    "original_gold_overlap": str(overlapping_gold),
                    "candidate_polity_prefix": cand["polity"],
                    "candidate_polity_span": f"{cand['start']}-{polity_end}",
                    "candidate_era_date_substring": cand["era"] + cand["numeral"] + "年",
                    "candidate_era_span": f"{era_start}-{cand['end']}",
                    "proposed_split": (
                        f"{cand['polity']}[{cand['start']}-{polity_end}]/ORG(?) + "
                        f"{cand['era']}{cand['numeral']}年[{era_start}-{cand['end']}]/DTM(?)"
                    ),
                    "exact_rule_match": "GR-11",
                    "confidence": conf,
                    "reviewer_decision": "",
                    "final_polity_label": "",
                    "final_era_label": "",
                    "reviewer_notes": "",
                })

    review_dir = os.path.join(args.out_dir, "review")
    os.makedirs(review_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(review_dir, "dynasty_era_candidates.xlsx"), index=False)

    conf_counts = df["confidence"].value_counts().to_dict() if len(df) else {}
    print(f"GR-11 candidates found: {len(rows)} (confidence breakdown: {conf_counts})")
    print(f"Output -> {review_dir}/dynasty_era_candidates.xlsx")
    for r in rows[:10]:
        print(f"  [{r['confidence']}] {r['split']}:{r['sample_id']} {r['marked_context']}")


if __name__ == "__main__":
    main()
