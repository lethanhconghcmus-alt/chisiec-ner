"""
annotation_audit.py — CLI chạy toàn bộ annotation consistency audit (mục
1-10 trong yêu cầu). CHỈ ĐỌC train/dev/test hiện có, KHÔNG sửa bất kỳ nhãn
nào, KHÔNG ghi đè dataset gốc. Output ghi vào artifacts/audit_v1/.

Usage:
  python scripts/annotation_audit.py \
      --train data/dvsk/dvsk_train.txt \
      --dev   data/dvsk/dvsk_dev.txt \
      --test  data/dvsk/dvsk_test.txt \
      --out-dir artifacts/audit_v1 \
      [--source-map path/to/record_source_map.json] \
      [--top-n 300] [--context-window 20]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import read_conll
from src.audit_utils import (
    CONFUSABLE_PAIRS,
    LOC_ADMIN_SUFFIXES,
    build_admin_suffix_boundary_report,
    build_ambiguity_report,
    build_confusable_pair_report,
    build_merge_split_report,
    build_review_workbook,
    extract_all_spans,
    load_source_map,
    _file_checksum,
)

MERGE_SPLIT_PAIRS = [("ORG", "TITLE"), ("LOC", "TITLE"), ("PER", "TITLE")]


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_csv(rows, path, columns=None):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, set)) else v)
                              for k, v in r.items()})


def build_summary_markdown(entities, ambiguity_report, admin_suffix_report,
                            merge_split_reports, confusable_pair_report,
                            review_rows, splits_data) -> str:
    lines = ["# Annotation Consistency Audit — Summary", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("**Lưu ý quan trọng**: report này chỉ liệt kê *candidate "
                  "inconsistency* (ứng viên cần adjudicate) — KHÔNG khẳng "
                  "định gold label sai. Quyết định cuối cùng thuộc về "
                  "reviewer con người theo guideline.")
    lines.append("")

    lines.append("## Số entity theo split và nhãn")
    lines.append("")
    lines.append("| split | " + " | ".join(sorted({e["label"] for e in entities})) + " | total |")
    lines.append("|---" * (len({e["label"] for e in entities}) + 2) + "|")
    for split in splits_data:
        counts = {}
        for e in entities:
            if e["split"] == split:
                counts[e["label"]] = counts.get(e["label"], 0) + 1
        labels_sorted = sorted({e["label"] for e in entities})
        row = [split] + [str(counts.get(l, 0)) for l in labels_sorted] + [str(sum(counts.values()))]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    surfaces_all = {e["surface"] for e in entities}
    lines.append(f"- Số surface forms unique (toàn bộ, chưa lọc theo label): **{len(surfaces_all)}**")
    lines.append(f"- Số ambiguous surfaces (>=2 nhãn khác nhau): **{len(ambiguity_report)}**")
    lines.append("")

    lines.append("## Top 50 ambiguous surfaces")
    lines.append("")
    lines.append("| surface | num_labels | total_occ | labels | splits |")
    lines.append("|---|---|---|---|---|")
    for r in ambiguity_report[:50]:
        lines.append(
            f"| {r['surface']} | {r['num_unique_labels']} | {r['total_occurrences']} | "
            f"{','.join(r['unique_labels'])} | {','.join(r['splits_present'])} |"
        )
    lines.append("")

    lines.append("## Top boundary inconsistencies (hậu tố hành chính LOC)")
    lines.append(f"Hậu tố xét: {' '.join(LOC_ADMIN_SUFFIXES)}")
    lines.append("")
    lines.append("| stem | with_suffix | without_suffix |")
    lines.append("|---|---|---|")
    for r in admin_suffix_report[:30]:
        lines.append(f"| {r['stem']} | {r['num_with_suffix']} | {r['num_without_suffix']} |")
    lines.append("")

    lines.append("## Merge/split inconsistency (candidate) theo cặp nhãn")
    lines.append("")
    for key, msr in merge_split_reports.items():
        lines.append(f"### {msr['type_a']} <-> {msr['type_b']}")
        lines.append(f"Số candidate: {len(msr['candidates'])}")
        for c in msr["candidates"][:10]:
            lines.append(
                f"- `{c['surface']}`: split_adjacent={c['split_adjacent_count']}, "
                f"split_untagged={c['split_untagged_count']}, "
                f"merged_forms={c['merged_forms']}"
            )
        lines.append("")

    lines.append("## Distribution độ dài entity theo nhãn")
    lines.append("")
    length_by_label = {}
    for e in entities:
        length_by_label.setdefault(e["label"], []).append(e["end"] - e["start"] + 1)
    lines.append("| label | count | mean_len | min | max |")
    lines.append("|---|---|---|---|---|")
    for label, lens in sorted(length_by_label.items()):
        lines.append(f"| {label} | {len(lens)} | {sum(lens) / len(lens):.2f} | {min(lens)} | {max(lens)} |")
    lines.append("")

    issue_counts = {}
    for row in review_rows:
        for it in row["issue_type"].split(";"):
            issue_counts[it] = issue_counts.get(it, 0) + 1
    lines.append("## Số case review theo issue type")
    lines.append("")
    for it, c in sorted(issue_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {it}: {c}")
    lines.append("")

    lines.append(f"## Đề xuất review")
    lines.append(f"Đã chọn **{len(review_rows)}** case ưu tiên cao nhất "
                  f"(xem `review/review_workbook.csv`/`.xlsx`) — mỗi dòng là "
                  f"1 candidate inconsistency cần adjudicate, KHÔNG phải "
                  f"khẳng định lỗi.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None,
                     help="Optional record_source_map.json (document_id join, best-effort)")
    ap.add_argument("--out-dir", default="artifacts/audit_v1")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--context-window", type=int, default=20)
    args = ap.parse_args()

    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=args.context_window)

    ambiguity_report = build_ambiguity_report(entities)
    admin_suffix_report = build_admin_suffix_boundary_report(entities)
    merge_split_reports = {
        f"{a}-{b}": build_merge_split_report(entities, a, b) for a, b in MERGE_SPLIT_PAIRS
    }
    confusable_pair_report = build_confusable_pair_report(entities, CONFUSABLE_PAIRS)

    review_rows = build_review_workbook(
        entities, ambiguity_report, admin_suffix_report, merge_split_reports,
        top_n=args.top_n, model_disagreements=None,  # xem README: chưa chạy model disagreement pass này
    )

    reports_dir = os.path.join(args.out_dir, "reports")
    review_dir = os.path.join(args.out_dir, "review")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    _save_jsonl(ambiguity_report, os.path.join(reports_dir, "ambiguity_report.jsonl"))
    _save_csv(
        [{"surface": r["surface"], "num_unique_labels": r["num_unique_labels"],
          "total_occurrences": r["total_occurrences"], "unique_labels": ",".join(r["unique_labels"]),
          "splits_present": ",".join(r["splits_present"])} for r in ambiguity_report],
        os.path.join(reports_dir, "ambiguity_report.csv"),
    )
    _save_json(admin_suffix_report, os.path.join(reports_dir, "boundary_admin_suffix_report.json"))
    _save_json(merge_split_reports, os.path.join(reports_dir, "merge_split_report.json"))
    _save_json(confusable_pair_report, os.path.join(reports_dir, "confusable_pair_report.json"))

    import pandas as pd
    from src.audit_utils import REVIEW_COLUMNS
    review_df = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    review_df.to_csv(os.path.join(review_dir, "review_workbook.csv"), index=False, encoding="utf-8-sig")
    review_df.to_excel(os.path.join(review_dir, "review_workbook.xlsx"), index=False)

    summary_md = build_summary_markdown(
        entities, ambiguity_report, admin_suffix_report, merge_split_reports,
        confusable_pair_report, review_rows, splits_data,
    )
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "dataset_checksums": {
            "train": _file_checksum(args.train),
            "dev": _file_checksum(args.dev),
            "test": _file_checksum(args.test),
        },
        "num_sentences": {k: len(v) for k, v in splits_data.items()},
        "num_entities": len(entities),
        "num_ambiguous_surfaces": len(ambiguity_report),
        "num_boundary_admin_suffix_stems": len(admin_suffix_report),
        "num_review_rows": len(review_rows),
        "model_disagreement_run": False,
    }
    _save_json(manifest, os.path.join(args.out_dir, "manifest.json"))

    print(f"Audit done. Entities={len(entities)}, ambiguous_surfaces={len(ambiguity_report)}, "
          f"review_rows={len(review_rows)}")
    print(f"Output -> {args.out_dir}")

    print("\n=== TOP 20 CANDIDATE INCONSISTENCIES ===")
    for row in review_rows[:20]:
        print(f"[{row['priority_score']:.2f}] {row['issue_type']} | {row['split']}#{row['sample_id']} | "
              f"gold='{row['gold_surface']}'({row['gold_label']}) | {row['marked_context']}")


if __name__ == "__main__":
    main()
