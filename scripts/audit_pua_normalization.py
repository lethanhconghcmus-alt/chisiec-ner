"""
audit_pua_normalization.py — quét PUA/variation-selector/replacement-char
trong toàn bộ train/dev/test, sinh pua_inventory.csv + dry-run impact report
(KHÔNG sửa dataset gốc).

Usage:
  python scripts/audit_pua_normalization.py \
      --train data/dvsk/dvsk_train.txt --dev ... --test ... \
      --normalization-config configs/normalization_v2.yaml \
      --out-dir artifacts/audit_v1
"""

import argparse
import csv
import json
import os
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data_utils import read_conll
from src.audit_utils import extract_all_spans, load_source_map
from src.normalization import codepoint_category, normalize_text_with_alignment


def load_verified_mapping(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mapping = {}
    for entry in cfg.get("verified_pua_mapping", []):
        cp = int(entry["codepoint"].replace("U+", ""), 16)
        mapping[cp] = entry["replacement"]
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None)
    ap.add_argument("--normalization-config", default="configs/normalization_v2.yaml")
    ap.add_argument("--out-dir", default="artifacts/audit_v1")
    args = ap.parse_args()

    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=20)
    verified_mapping = load_verified_mapping(args.normalization_config)

    # ── 1. Scan mọi ký tự trong mọi câu (KHÔNG chỉ trong entity) ──────────
    cp_records = defaultdict(lambda: {
        "count": 0, "sample_refs": [], "contexts": [],
        "inside_gold_entity_count": 0, "entity_examples": set(),
    })

    entity_char_index = defaultdict(list)  # (split, sample_id) -> [(start,end,label,surface)]
    for e in entities:
        entity_char_index[(e["split"], e["sample_id"])].append(
            (e["start"], e["end"], e["label"], e["surface"])
        )

    for split, sentences in splits_data.items():
        for idx, (tokens, labels) in enumerate(sentences):
            text = "".join(tokens)
            ents_here = entity_char_index.get((split, idx), [])
            for pos, ch in enumerate(text):
                cp = ord(ch)
                cat = codepoint_category(cp)
                if cat is None:
                    continue
                rec = cp_records[cp]
                rec["count"] += 1
                if len(rec["sample_refs"]) < 20:
                    rec["sample_refs"].append(f"{split}:{idx}")
                if len(rec["contexts"]) < 10:
                    ctx_s, ctx_e = max(0, pos - 20), min(len(text), pos + 21)
                    rec["contexts"].append(text[ctx_s:pos] + f"[{ch}]" + text[pos + 1:ctx_e])
                for (es, ee, label, surface) in ents_here:
                    if es <= pos <= ee:
                        rec["inside_gold_entity_count"] += 1
                        if len(rec["entity_examples"]) < 10:
                            rec["entity_examples"].add(f"{surface}/{label}")

    inventory_rows = []
    for cp, rec in sorted(cp_records.items(), key=lambda kv: -kv[1]["count"]):
        cat = codepoint_category(cp)
        try:
            name = unicodedata.name(chr(cp))
        except ValueError:
            name = ""
        mapped = verified_mapping.get(cp)
        inventory_rows.append({
            "codepoint": f"U+{cp:04X}",
            "category": cat,
            "unicode_name_if_available": name,
            "count": rec["count"],
            "sample_ids": ";".join(rec["sample_refs"]),
            "contexts": " | ".join(rec["contexts"]),
            "inside_gold_entity_count": rec["inside_gold_entity_count"],
            "entity_examples": ";".join(sorted(rec["entity_examples"])),
            "proposed_mapping": mapped or "",
            "mapping_status": "verified" if mapped else "unmapped",
        })

    reports_dir = os.path.join(args.out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "pua_inventory.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory_rows[0].keys()) if inventory_rows else [])
        writer.writeheader()
        for r in inventory_rows:
            writer.writerow(r)

    # ── 2. Verify context riêng cho U+F0C51 / U+F025F ─────────────────────
    verify_detail = {}
    for cp_hex in ("U+F0C51", "U+F025F"):
        cp = int(cp_hex.replace("U+", ""), 16)
        rec = cp_records.get(cp)
        verify_detail[cp_hex] = {
            "count": rec["count"] if rec else 0,
            "all_contexts": rec["contexts"] if rec else [],
            "entity_examples": sorted(rec["entity_examples"]) if rec else [],
        }

    # ── 3. Dry-run impact: normalize toàn bộ câu, đếm ảnh hưởng ───────────
    n_affected_records = 0
    n_affected_gold_spans = 0
    affected_labels = defaultdict(int)
    n_spans_offset_changed = 0
    yushitai_report = []

    for split, sentences in splits_data.items():
        for idx, (tokens, labels) in enumerate(sentences):
            text = "".join(tokens)
            normalized_text, raw_to_norm, norm_to_raw, applied = normalize_text_with_alignment(
                text, mapping=verified_mapping,
            )
            if not applied:
                continue
            n_affected_records += 1
            ents_here = entity_char_index.get((split, idx), [])
            for (es, ee, label, surface) in ents_here:
                touches_pua = any(es <= r["raw_index"] <= ee for r in applied)
                if not touches_pua:
                    continue
                n_affected_gold_spans += 1
                affected_labels[label] += 1
                from src.normalization import remap_span_inclusive
                n_start, n_end = remap_span_inclusive(raw_to_norm, es, ee, len(text), len(normalized_text))
                new_surface = normalized_text[n_start:n_end + 1]
                if (n_start, n_end) != (es, ee):
                    n_spans_offset_changed += 1
                if surface in ("御史", "御史臺") or new_surface in ("御史", "御史臺"):
                    yushitai_report.append({
                        "split": split, "sample_id": idx, "label": label,
                        "raw_surface": surface, "normalized_surface": new_surface,
                        "raw_span": [es, ee], "normalized_span": [n_start, n_end],
                    })

    dry_run_report = {
        "total_pua_chars_all_categories": sum(r["count"] for r in inventory_rows),
        "total_verified_pua_chars": sum(r["count"] for r in inventory_rows if r["mapping_status"] == "verified"),
        "total_unmapped_suspicious_chars": sum(r["count"] for r in inventory_rows if r["mapping_status"] == "unmapped"),
        "num_affected_records": n_affected_records,
        "num_affected_gold_spans": n_affected_gold_spans,
        "affected_labels": dict(affected_labels),
        "num_spans_with_offset_change_after_normalization": n_spans_offset_changed,
        "note_on_offset_change": (
            "Mapping U+F0C51/U+F025F -> 臺 la 1 ky tu doi 1 ky tu (length-"
            "preserving) nen offset KHONG doi -- 0 la ket qua ky vong, "
            "khong phai loi. Viec mo rong span 御史->御史臺 la 1 hanh dong "
            "CORRECT_BOUNDARY rieng biet (GR-02 phan 2), KHONG phai do "
            "normalization gay ra."
        ),
        "verify_detail_F0C51_F025F": verify_detail,
        "yushitai_cases": yushitai_report,
    }
    with open(os.path.join(reports_dir, "pua_normalization_dry_run.json"), "w", encoding="utf-8") as f:
        json.dump(dry_run_report, f, ensure_ascii=False, indent=2)

    print(f"PUA inventory: {len(inventory_rows)} distinct codepoint(s) -> reports/pua_inventory.csv")
    print(f"Verified mapped chars: {dry_run_report['total_verified_pua_chars']}")
    print(f"Unmapped suspicious chars: {dry_run_report['total_unmapped_suspicious_chars']}")
    print(f"Affected records: {n_affected_records}, affected gold spans: {n_affected_gold_spans}")
    print(f"Affected labels: {dict(affected_labels)}")
    print(f"Spans with offset change: {n_spans_offset_changed} (kỳ vọng 0)")
    print(f"御史/御史臺 cases found: {len(yushitai_report)}")
    for c in yushitai_report:
        print(f"  {c}")


if __name__ == "__main__":
    main()
