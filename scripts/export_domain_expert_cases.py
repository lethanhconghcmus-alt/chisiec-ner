"""
export_domain_expert_cases.py — xuất riêng các occurrence có
reviewer_decision=NEEDS_DOMAIN_EXPERT (mục A.1: 4 case 日燏) với context
±50, full metadata. Đọc-only, không sửa gì.

Usage:
  python scripts/export_domain_expert_cases.py \
      --train ... --dev ... --test ... \
      --guideline-workbook artifacts/audit_v1/review/review_guideline_ambiguities.xlsx \
      --out artifacts/audit_v1/review/domain_expert_cases_riyue.xlsx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_utils import read_conll
from src.audit_utils import extract_all_spans, get_neighboring_entities, load_source_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None)
    ap.add_argument("--guideline-workbook", required=True)
    ap.add_argument("--out", default="artifacts/audit_v1/review/domain_expert_cases_riyue.xlsx")
    args = ap.parse_args()

    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=50)
    entity_by_id = {e["entity_id"]: e for e in entities}

    df = pd.read_excel(args.guideline_workbook, sheet_name="deepdive_occurrences")
    pending = df[(df["surface"] == "日燏") & (df["reviewer_decision"] == "NEEDS_DOMAIN_EXPERT")]

    rows = []
    for _, r in pending.iterrows():
        e = entity_by_id.get(r["entity_id"])
        if e is None:
            continue
        neighbors = get_neighboring_entities(entities, e, window=50)
        ctx_s = max(0, e["start"] - 50)
        ctx_e = min(len(e["text"]), e["end"] + 1 + 50)
        marked = f"{e['text'][ctx_s:e['start']]}【{e['surface']}】{e['text'][e['end']+1:ctx_e]}"
        rows.append({
            "entity_id": e["entity_id"],
            "split": e["split"],
            "sample_id": e["sample_id"],
            "document_id": e.get("document_id"),
            "raw_gold_label_v1": e["label"],
            "gold_surface": e["surface"],
            "gold_start": e["start"],
            "gold_end": e["end"],
            "full_sentence": e["text"],
            "context_pm50": marked,
            "neighboring_entities": "; ".join(f"{n['surface']}/{n['label']}" for n in neighbors),
            "reviewer_notes_from_deepdive": r.get("reviewer_notes", ""),
            "apply_to_dataset_v2": False,
            "status": "unresolved_pending_domain_expert",
            "domain_expert_final_label": "",
            "domain_expert_final_start": "",
            "domain_expert_final_end": "",
            "domain_expert_notes": "",
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_excel(args.out, index=False)
    print(f"Exported {len(rows)} domain-expert-pending case(s) -> {args.out}")
    for r in rows:
        print(f"  {r['split']}:{r['sample_id']} raw_label={r['raw_gold_label_v1']} {r['context_pm50']}")


if __name__ == "__main__":
    main()
