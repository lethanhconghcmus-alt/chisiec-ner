"""
apply_adjudication.py — đọc dataset v1 (immutable) + review workbook đã
điền, validate nghiêm ngặt, sinh proposed change. Mặc định LUÔN dry-run.
Chỉ ghi dataset_v2 thật khi truyền --apply TƯỜNG MINH.

Usage (dry-run, mặc định):
  python scripts/apply_adjudication.py \
      --train data/dvsk/dvsk_train.txt --dev ... --test ... \
      --review-dir artifacts/audit_v1/review \
      --audit-manifest artifacts/audit_v1/manifest.json \
      --out-dir artifacts/adjudication_v2_dry_run

Chỉ khi đã xem kỹ dry-run report và xác nhận, chạy lại với:
  python scripts/apply_adjudication.py ... --apply --out-dir data/dataset_v2
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_utils import read_conll
from src.audit_utils import extract_all_spans, load_source_map, _file_checksum
from src.adjudication import (
    build_entity_index,
    process_boundary_split_fillin,
    process_collision_candidates,
    process_deepdive_occurrences,
    process_guideline_ambiguities_surface_level,
    process_singleton_anomalies,
    validate_dataset_checksums,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None)
    ap.add_argument("--review-dir", required=True,
                     help="Thư mục chứa review_priority_singleton_anomalies.xlsx và review_guideline_ambiguities.xlsx")
    ap.add_argument("--audit-manifest", default=None,
                     help="artifacts/audit_v1/manifest.json gốc, dùng để so checksum dataset")
    ap.add_argument("--boundary-split-fillin", default=None,
                     help="boundary_split_fill_in.xlsx (scripts/export_boundary_split_fillin.py) đã điền final_start/final_end, optional")
    ap.add_argument("--collision-candidates", default=None,
                     help="collision_merge_candidates.xlsx (scripts/export_collision_candidates.py), optional")
    ap.add_argument("--out-dir", default="artifacts/adjudication_v2_dry_run")
    ap.add_argument("--apply", action="store_true",
                     help="BẮT BUỘC truyền tường minh để ghi dataset_v2 thật. Mặc định KHÔNG có = dry-run.")
    args = ap.parse_args()
    dry_run = not args.apply

    # ── 1. Đọc dataset v1 (immutable) ───────────────────────────────────
    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    current_checksums = {
        "train": _file_checksum(args.train),
        "dev": _file_checksum(args.dev),
        "test": _file_checksum(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=20)
    entity_index = build_entity_index(entities)

    # ── Global checksum validation ───────────────────────────────────────
    global_errors = []
    if args.audit_manifest and os.path.exists(args.audit_manifest):
        with open(args.audit_manifest, encoding="utf-8") as f:
            prior_manifest = json.load(f)
        global_errors.extend(validate_dataset_checksums(prior_manifest, current_checksums))
    else:
        global_errors.append(
            "Khong tim thay --audit-manifest -- KHONG the xac minh dataset "
            "hien tai co giong luc workbook duoc review hay khong."
        )

    # ── 2. Đọc review workbook decisions ────────────────────────────────
    singleton_path = os.path.join(args.review_dir, "review_priority_singleton_anomalies.xlsx")
    guideline_path = os.path.join(args.review_dir, "review_guideline_ambiguities.xlsx")

    all_proposed, all_skipped = [], []

    if os.path.exists(singleton_path):
        df_singleton = pd.read_excel(singleton_path)
        proposed, skipped = process_singleton_anomalies(df_singleton)
        all_proposed.extend(proposed)
        all_skipped.extend(skipped)
    else:
        global_errors.append(f"Khong tim thay {singleton_path}")

    if os.path.exists(guideline_path):
        df_surface = pd.read_excel(guideline_path, sheet_name="guideline_ambiguities")
        all_skipped.extend(process_guideline_ambiguities_surface_level(df_surface))

        df_deep = pd.read_excel(guideline_path, sheet_name="deepdive_occurrences")
        proposed, skipped = process_deepdive_occurrences(df_deep, entity_index,
                                                           change_id_start=len(all_proposed))
        all_proposed.extend(proposed)
        all_skipped.extend(skipped)
    else:
        global_errors.append(f"Khong tim thay {guideline_path}")

    # Các case đã được xác định là "collision" (xem
    # scripts/export_collision_candidates.py) PHẢI được xử lý qua
    # collision_merge_candidates.xlsx (phân loại riêng, không tự merge),
    # KHÔNG được xử lý lại như CORRECT_BOUNDARY thường qua
    # boundary_split_fill_in.xlsx (tránh báo trùng 1 case ở cả 2 nơi).
    collision_keys = set()
    if args.collision_candidates and os.path.exists(args.collision_candidates):
        df_collision = pd.read_excel(args.collision_candidates)
        for _, r in df_collision.iterrows():
            ps, pe = str(r["proposed_new_span"]).split("-")
            collision_keys.add((r["split"], int(r["sample_id"]), int(ps), int(pe)))

    if args.boundary_split_fillin:
        if os.path.exists(args.boundary_split_fillin):
            df_bs = pd.read_excel(args.boundary_split_fillin)
            if collision_keys:
                is_collision = df_bs.apply(
                    lambda r: (r["split"], int(r["sample_id"]), int(r["final_start"]), int(r["final_end"]))
                    in collision_keys, axis=1,
                )
                n_excluded = int(is_collision.sum())
                if n_excluded:
                    print(f"[INFO] Loai {n_excluded} dong khoi boundary_split_fillin vi da "
                          f"duoc xu ly rieng qua collision_merge_candidates.xlsx")
                df_bs = df_bs[~is_collision]
            proposed, skipped = process_boundary_split_fillin(df_bs, entity_index,
                                                                change_id_start=len(all_proposed))
            all_proposed.extend(proposed)
            all_skipped.extend(skipped)
        else:
            global_errors.append(f"Khong tim thay {args.boundary_split_fillin}")

    if args.collision_candidates:
        if os.path.exists(args.collision_candidates):
            proposed, skipped = process_collision_candidates(
                df_collision, entities, entity_index, change_id_start=len(all_proposed),
            )
            all_proposed.extend(proposed)
            all_skipped.extend(skipped)
        else:
            global_errors.append(f"Khong tim thay {args.collision_candidates}")

    # ── 3. Phân loại valid / error ───────────────────────────────────────
    valid_changes = [c for c in all_proposed if c.validation_status == "valid"]
    error_changes = [c for c in all_proposed if c.validation_status != "valid"]

    # ── 4. Ghi output (luôn ghi, kể cả dry-run) ─────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "proposed_dataset_diff.jsonl"), "w", encoding="utf-8") as f:
        for c in valid_changes:
            f.write(json.dumps(c.to_row(), ensure_ascii=False) + "\n")

    changelog_rows = [c.to_row() for c in all_proposed]
    with open(os.path.join(args.out_dir, "changelog_draft.csv"), "w", newline="", encoding="utf-8-sig") as f:
        if changelog_rows:
            writer = csv.DictWriter(f, fieldnames=list(changelog_rows[0].keys()))
            writer.writeheader()
            for r in changelog_rows:
                writer.writerow(r)

    with open(os.path.join(args.out_dir, "skipped_decisions.csv"), "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source_workbook_row", "split", "sample_id", "surface", "reviewer_decision", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_skipped:
            writer.writerow({
                "source_workbook_row": s.source_workbook_row, "split": s.split,
                "sample_id": s.sample_id, "surface": s.surface,
                "reviewer_decision": s.reviewer_decision, "reason": s.reason,
            })

    with open(os.path.join(args.out_dir, "validation_errors.csv"), "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["change_id", "split", "sample_id", "surface_old_label", "new_label",
                      "source_workbook_row", "validation_message"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in error_changes:
            writer.writerow({
                "change_id": c.change_id, "split": c.split, "sample_id": c.sample_id,
                "surface_old_label": c.old_label, "new_label": c.new_label,
                "source_workbook_row": c.source_workbook_row,
                "validation_message": c.validation_message,
            })
        for i, ge in enumerate(global_errors):
            writer.writerow({
                "change_id": f"GLOBAL-{i}", "split": "", "sample_id": "",
                "surface_old_label": "", "new_label": "",
                "source_workbook_row": "GLOBAL", "validation_message": ge,
            })

    decision_counts = {}
    for s in all_skipped:
        decision_counts.setdefault(s.reviewer_decision or "(empty)", {}).setdefault(s.reason, 0)
        decision_counts[s.reviewer_decision or "(empty)"][s.reason] += 1

    summary_lines = [
        "# Adjudication dry-run summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Mode: {'DRY RUN (mac dinh)' if dry_run else 'APPLY'}",
        "",
        f"- Proposed changes (valid, sẵn sàng apply): **{len(valid_changes)}**",
        f"- Proposed changes (validation error, KHÔNG áp): **{len(error_changes)}**",
        f"- Skipped decisions (theo policy, không cố gắng apply): **{len(all_skipped)}**",
        f"- Global validation errors: **{len(global_errors)}**",
        "",
        "## Global errors",
    ]
    for ge in global_errors:
        summary_lines.append(f"- {ge}")
    summary_lines.append("")
    summary_lines.append("## Skipped theo lý do")
    reason_counts = {}
    for s in all_skipped:
        reason_counts[s.reason] = reason_counts.get(s.reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        summary_lines.append(f"- {reason}: {count}")
    summary_lines.append("")
    summary_lines.append("## Valid proposed changes (mẫu, tối đa 30 dòng)")
    for c in valid_changes[:30]:
        summary_lines.append(
            f"- [{c.split}:{c.sample_id}] {c.old_label} -> {c.new_label} "
            f"(span {c.old_span} không đổi, decision={c.reviewer_decision})"
        )
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # ── 5. Chỉ ghi dataset_v2 thật nếu --apply ──────────────────────────
    if args.apply:
        raise SystemExit(
            "--apply duoc truyen nhung script nay CHUA implement buoc ghi "
            "dataset_v2 that (materialize) -- theo dung yeu cau hien tai "
            "chi lam den muc dry-run + validate. Dung lai o day, KHONG ghi "
            "gi vao data/dataset_v2/."
        )

    print(f"[DRY RUN] Valid proposed changes: {len(valid_changes)}")
    print(f"[DRY RUN] Validation errors: {len(error_changes)}")
    print(f"[DRY RUN] Skipped decisions: {len(all_skipped)}")
    print(f"[DRY RUN] Global errors: {len(global_errors)}")
    print(f"Output -> {args.out_dir}/")
    print("\nDecision x reason breakdown:")
    for decision, reasons in decision_counts.items():
        print(f"  {decision}:")
        for reason, count in reasons.items():
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
