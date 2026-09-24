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
import shutil
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_utils import read_conll
from src.audit_utils import extract_all_spans, load_source_map, _file_checksum
from src.materialize import materialize_dataset_v2, utc_now_iso
from src.adjudication import (
    ProposedMerge,
    ProposedRemoval,
    build_entity_index,
    process_boundary_split_fillin,
    process_collision_candidates,
    process_deepdive_occurrences,
    process_gr11_date_formula_candidates,
    process_guideline_ambiguities_surface_level,
    process_review_transactions,
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
    ap.add_argument("--transactions", action="append", default=[],
                     help="File transaction (vd yushi_gudu_transaction.xlsx) — mỗi dòng 1 action, "
                          "cột transaction_id nhóm action commit atomic. Có thể truyền nhiều lần. "
                          "CHỈ truyền khi reviewer đã điền đủ field bắt buộc.")
    ap.add_argument("--gr11-candidates", default=None,
                     help="gr11_date_formula_candidates.xlsx (scripts/audit_date_formula_gr11.py) "
                          "đã review đủ reviewer_decision. GR-11b (dynasty_era_candidates.xlsx, "
                          "scan_dynasty_era_candidates.py, policy cũ) KHÔNG có flag riêng -- "
                          "cố tình deferred, không đọc trong script này.")
    ap.add_argument("--out-dir", default="artifacts/adjudication_v2_dry_run")
    ap.add_argument("--apply", action="store_true",
                     help="BẮT BUỘC truyền tường minh để ghi dataset_v2 thật. Mặc định KHÔNG có = dry-run.")
    ap.add_argument("--force-overwrite", action="store_true",
                     help="CHỈ có tác dụng cùng --apply. Cho phép overwrite --out-dir đã tồn tại "
                          "(bản cũ được rename sang backup timestamped, KHÔNG xoá).")
    ap.add_argument("--expect-changes", type=int, default=None,
                     help="Contract gate (chỉ dùng cùng --apply): số valid direct changes ky vong "
                          "(review-level, truoc materializer dedup). Neu khac -> tu choi apply.")
    ap.add_argument("--expect-merges", type=int, default=None,
                     help="Contract gate: so valid merges ky vong (review-level).")
    ap.add_argument("--expect-removals", type=int, default=None,
                     help="Contract gate: so valid removals ky vong.")
    ap.add_argument("--expect-quarantine-rows", type=int, default=None,
                     help="Contract gate: so quarantine DECISION ROW ky vong (khong phai unique sample).")
    ap.add_argument("--expect-quarantine-unique", type=int, default=None,
                     help="Contract gate: so quarantine UNIQUE sample_id ky vong (se bi loai khoi v2_clean).")
    ap.add_argument("--expect-total-sentences", type=int, default=None,
                     help="Contract gate: tong so cau (train+dev+test) ky vong trong dataset_v2_full.")
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
    # GR-11b (dynasty_era_candidates.xlsx, scan_dynasty_era_candidates.py, policy
    # GR-11 CU) khong co CLI flag rieng va khong bao gio duoc doc trong script nay
    # -- checksum truoc/sau chi de assert tuong minh "thuc su khong dong vao".
    gr11b_path = os.path.join(args.review_dir, "dynasty_era_candidates.xlsx")
    gr11b_checksum_before = _file_checksum(gr11b_path) if os.path.exists(gr11b_path) else None
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

    all_proposed, all_skipped, all_quarantined = [], [], []

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

    for tx_path in args.transactions:
        if os.path.exists(tx_path):
            df_tx = pd.read_excel(tx_path)
            proposed, skipped = process_review_transactions(
                df_tx, entities, entity_index, change_id_start=len(all_proposed),
            )
            all_proposed.extend(proposed)
            all_skipped.extend(skipped)
        else:
            global_errors.append(f"Khong tim thay {tx_path}")

    if args.gr11_candidates:
        if os.path.exists(args.gr11_candidates):
            df_gr11 = pd.read_excel(args.gr11_candidates)
            proposed, skipped, quarantined = process_gr11_date_formula_candidates(
                df_gr11, entities, entity_index, change_id_start=len(all_proposed),
            )
            all_proposed.extend(proposed)
            all_skipped.extend(skipped)
            all_quarantined.extend(quarantined)
        else:
            global_errors.append(f"Khong tim thay {args.gr11_candidates}")

    # ── 3. Phân loại valid / error (tách riêng ProposedChange vs ProposedMerge
    # vs ProposedRemoval -- schema khác nhau, không gộp chung 1 CSV) ──────
    all_removals = [c for c in all_proposed if isinstance(c, ProposedRemoval)]
    all_changes = [c for c in all_proposed if not isinstance(c, ProposedMerge) and not isinstance(c, ProposedRemoval)]
    all_merges = [c for c in all_proposed if isinstance(c, ProposedMerge)]

    valid_removals = [r for r in all_removals if r.validation_status == "valid"]
    error_removals = [r for r in all_removals if r.validation_status != "valid"]

    valid_changes = [c for c in all_changes if c.validation_status == "valid"]
    error_changes = [c for c in all_changes if c.validation_status != "valid"]
    valid_merges = [m for m in all_merges if m.validation_status == "valid"]
    error_merges = [m for m in all_merges if m.validation_status != "valid"]

    # ── 4. Ghi report (dry-run: vao thang --out-dir; apply: vao 1 thu muc
    # tam rieng CHI de kiem tra validation_errors truoc khi materialize --
    # --out-dir khi --apply la DICH CUOI CUNG cua dataset that, KHONG duoc
    # pre-tao no o day keo apply-gate ben duoi luon thay "da ton tai") ────
    if args.apply:
        import tempfile
        report_out_dir = tempfile.mkdtemp(prefix="adjudication_apply_prevalidate_")
    else:
        report_out_dir = args.out_dir
    os.makedirs(report_out_dir, exist_ok=True)

    with open(os.path.join(report_out_dir, "proposed_dataset_diff.jsonl"), "w", encoding="utf-8") as f:
        for c in valid_changes:
            f.write(json.dumps(c.to_row(), ensure_ascii=False) + "\n")

    with open(os.path.join(report_out_dir, "proposed_merges_diff.jsonl"), "w", encoding="utf-8") as f:
        for m in valid_merges:
            f.write(json.dumps(m.to_row(), ensure_ascii=False) + "\n")

    changelog_rows = [c.to_row() for c in all_changes]
    with open(os.path.join(report_out_dir, "changelog_draft.csv"), "w", newline="", encoding="utf-8-sig") as f:
        if changelog_rows:
            writer = csv.DictWriter(f, fieldnames=list(changelog_rows[0].keys()))
            writer.writeheader()
            for r in changelog_rows:
                writer.writerow(r)

    merge_rows = [m.to_row() for m in all_merges]
    with open(os.path.join(report_out_dir, "merge_changelog_draft.csv"), "w", newline="", encoding="utf-8-sig") as f:
        if merge_rows:
            writer = csv.DictWriter(f, fieldnames=list(merge_rows[0].keys()))
            writer.writeheader()
            for r in merge_rows:
                writer.writerow(r)

    removal_rows = [r.to_row() for r in all_removals]
    with open(os.path.join(report_out_dir, "removal_changelog_draft.csv"), "w", newline="", encoding="utf-8-sig") as f:
        if removal_rows:
            writer = csv.DictWriter(f, fieldnames=list(removal_rows[0].keys()))
            writer.writeheader()
            for r in removal_rows:
                writer.writerow(r)

    with open(os.path.join(report_out_dir, "skipped_decisions.csv"), "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source_workbook_row", "split", "sample_id", "surface", "reviewer_decision", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_skipped:
            writer.writerow({
                "source_workbook_row": s.source_workbook_row, "split": s.split,
                "sample_id": s.sample_id, "surface": s.surface,
                "reviewer_decision": s.reviewer_decision, "reason": s.reason,
            })

    with open(os.path.join(report_out_dir, "quarantine_decisions.csv"), "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source_workbook_row", "split", "sample_id", "surface", "reviewer_decision", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in all_quarantined:
            writer.writerow({
                "source_workbook_row": q.source_workbook_row, "split": q.split,
                "sample_id": q.sample_id, "surface": q.surface,
                "reviewer_decision": q.reviewer_decision, "reason": q.reason,
            })

    with open(os.path.join(report_out_dir, "validation_errors.csv"), "w", newline="", encoding="utf-8-sig") as f:
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
        for m in error_merges:
            writer.writerow({
                "change_id": m.change_id, "split": m.split, "sample_id": m.sample_id,
                "surface_old_label": "+".join(s[3] for s in m.source_spans),
                "new_label": m.resulting_label,
                "source_workbook_row": m.source_workbook_row,
                "validation_message": m.validation_message,
            })
        for r in error_removals:
            writer.writerow({
                "change_id": r.change_id, "split": r.split, "sample_id": r.sample_id,
                "surface_old_label": r.old_label, "new_label": "REMOVED",
                "source_workbook_row": r.source_workbook_row,
                "validation_message": r.validation_message,
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
        f"- Proposed merges (valid, sẵn sàng apply): **{len(valid_merges)}**",
        f"- Proposed merges (validation error, KHÔNG áp): **{len(error_merges)}**",
        f"- Proposed removals (valid, sẵn sàng apply): **{len(valid_removals)}**",
        f"- Proposed removals (validation error, KHÔNG áp): **{len(error_removals)}**",
        f"- Skipped decisions (theo policy, không cố gắng apply): **{len(all_skipped)}**",
        f"- Quarantined (SKIP_UNANNOTATED_SAMPLE, chờ quarantine_v2): **{len(all_quarantined)}**",
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
    if valid_merges:
        summary_lines.append("")
        summary_lines.append("## Valid proposed merges")
        for m in valid_merges:
            src_desc = " + ".join(f"{s[3]}/{s[2]}" for s in m.source_spans)
            summary_lines.append(
                f"- [{m.split}:{m.sample_id}] {src_desc} -> "
                f"{m.resulting_label}[{m.resulting_span[0]}-{m.resulting_span[1]}] "
                f"(rule={m.rule_id})"
            )
    with open(os.path.join(report_out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    if args.apply:
        # report_out_dir chi la scratch de kiem tra validation truoc khi
        # materialize -- changelog that nam trong final_out_dir/changelog/.
        shutil.rmtree(report_out_dir, ignore_errors=True)

    # ── 5. Chỉ ghi dataset_v2 thật nếu --apply ──────────────────────────
    if args.apply:
        if error_changes or error_merges or error_removals or global_errors:
            print("[APPLY] TU CHOI: van con validation error hoac global error, "
                  "sua het roi chay lai dry-run truoc.", file=sys.stderr)
            print(f"  error_changes={len(error_changes)} error_merges={len(error_merges)} "
                  f"error_removals={len(error_removals)} global_errors={len(global_errors)}", file=sys.stderr)
            raise SystemExit(1)

        from src.materialize import quarantine_sample_ids_by_split
        quarantine_ids_preview = quarantine_sample_ids_by_split(all_quarantined)
        quarantine_unique_preview = sum(len(v) for v in quarantine_ids_preview.values())
        contract_checks = [
            ("valid direct changes", args.expect_changes, len(valid_changes)),
            ("valid merges", args.expect_merges, len(valid_merges)),
            ("valid removals", args.expect_removals, len(valid_removals)),
            ("quarantine decision rows", args.expect_quarantine_rows, len(all_quarantined)),
            ("quarantine unique records", args.expect_quarantine_unique, quarantine_unique_preview),
        ]
        contract_failures = [f"{name}: expected {exp}, got {actual}"
                              for name, exp, actual in contract_checks if exp is not None and exp != actual]
        if contract_failures:
            print("[APPLY] TU CHOI: contract gate KHONG khop so lieu da duoc duyet -- "
                  "dataset hoac review workbook co the da doi kem tu luc duyet baseline.", file=sys.stderr)
            for f_ in contract_failures:
                print(f"  - {f_}", file=sys.stderr)
            raise SystemExit(1)

        final_out_dir = os.path.abspath(args.out_dir)
        if os.path.exists(final_out_dir):
            if not args.force_overwrite:
                print(f"[APPLY] TU CHOI: --out-dir {final_out_dir} da ton tai. "
                      f"Truyen --force-overwrite neu muon ghi de (ban cu se duoc "
                      f"rename sang backup timestamped, KHONG bi xoa).", file=sys.stderr)
                raise SystemExit(1)
            ts_backup = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = f"{final_out_dir}.backup_{ts_backup}"
            shutil.move(final_out_dir, backup_dir)
            print(f"[APPLY] Da rename output cu sang backup: {backup_dir}")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        temp_root = os.path.join(
            os.path.dirname(final_out_dir) or ".",
            f".{os.path.basename(final_out_dir)}_tmp_{ts}_{uuid.uuid4().hex[:8]}",
        )

        applied_review_workbooks = []
        for p in [singleton_path, guideline_path, args.boundary_split_fillin,
                  args.collision_candidates, args.gr11_candidates, *args.transactions]:
            if p and os.path.exists(p):
                applied_review_workbooks.append({"path": p, "checksum_md5": _file_checksum(p)})

        materialization_config = {
            "cli_args": vars(args), "generated_at_utc": utc_now_iso(),
            "dry_run_valid_counts": {
                "changes": len(valid_changes), "merges": len(valid_merges),
                "removals": len(valid_removals), "skipped": len(all_skipped),
                "quarantined": len(all_quarantined),
            },
        }
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        guideline_source_yaml = os.path.join(repo_dir, "docs", "guideline_v2.0_draft.yaml")

        print(f"[APPLY] Ghi vao temp dir: {temp_root}")
        success, mat_errors, report = materialize_dataset_v2(
            temp_root=temp_root, splits_v1=splits_data, source_map=source_map,
            valid_changes=valid_changes, valid_merges=valid_merges, valid_removals=valid_removals,
            all_skipped=all_skipped, all_quarantined=all_quarantined,
            materialization_config=materialization_config, repo_dir=repo_dir,
            audit_manifest_path=args.audit_manifest, current_checksums=current_checksums,
            applied_review_workbooks=applied_review_workbooks,
            guideline_source_yaml=guideline_source_yaml,
            expected_quarantine_unique=args.expect_quarantine_unique,
            expected_total_sentences=args.expect_total_sentences,
        )

        if not success:
            print("[APPLY] THAT BAI post-write validation -- KHONG rename temp thanh output cuoi. "
                  "Temp dir GIU LAI de debug:", temp_root, file=sys.stderr)
            for e in mat_errors[:50]:
                print(f"  - {e}", file=sys.stderr)
            raise SystemExit(1)

        # ── Immutability guard cuoi cung (source v1 + GR-11b deferred) ────
        from src.materialize import verify_source_files_unchanged
        immut_errors = verify_source_files_unchanged(
            {"train": args.train, "dev": args.dev, "test": args.test}, current_checksums,
        )
        if os.path.exists(gr11b_path):
            gr11b_after = _file_checksum(gr11b_path)
            if gr11b_checksum_before is not None and gr11b_after != gr11b_checksum_before:
                immut_errors.append(
                    f"GR-11b file {gr11b_path} DA BI THAY DOI sau materialize "
                    f"({gr11b_checksum_before} -> {gr11b_after})"
                )
        if immut_errors:
            print("[APPLY] THAT BAI immutability guard (source v1 hoac GR-11b bi doi) -- "
                  "KHONG rename temp thanh output cuoi. Temp dir GIU LAI de debug:",
                  temp_root, file=sys.stderr)
            for e in immut_errors:
                print(f"  - {e}", file=sys.stderr)
            raise SystemExit(1)

        os.rename(temp_root, final_out_dir)
        with open(os.path.join(final_out_dir, "MATERIALIZATION_SUCCESS.json"), "w", encoding="utf-8") as f:
            json.dump({
                "success": True, "materialized_at_utc": utc_now_iso(),
                "final_out_dir": final_out_dir, "report_summary": {
                    "record_counts": report.get("record_counts"),
                    "applied_counts": report.get("applied_counts"),
                    "quarantine_decision_rows_total": report.get("quarantine_decision_rows_total"),
                    "quarantined_unique_records_total": report.get("quarantined_unique_records_total"),
                },
            }, f, ensure_ascii=False, indent=2)

        print(f"[APPLY] THANH CONG. Output final: {final_out_dir}")
        print(f"[APPLY] Applied changes={len(valid_changes)} merges={len(valid_merges)} "
              f"removals={len(valid_removals)} quarantined={len(all_quarantined)}")
        return

    print(f"[DRY RUN] Valid proposed changes: {len(valid_changes)}")
    print(f"[DRY RUN] Validation errors (changes): {len(error_changes)}")
    print(f"[DRY RUN] Valid proposed merges: {len(valid_merges)}")
    print(f"[DRY RUN] Validation errors (merges): {len(error_merges)}")
    print(f"[DRY RUN] Valid proposed removals: {len(valid_removals)}")
    print(f"[DRY RUN] Validation errors (removals): {len(error_removals)}")
    print(f"[DRY RUN] Skipped decisions: {len(all_skipped)}")
    print(f"[DRY RUN] Quarantined (SKIP_UNANNOTATED_SAMPLE): {len(all_quarantined)}")
    print(f"[DRY RUN] Global errors: {len(global_errors)}")
    print(f"Output -> {args.out_dir}/")
    print("\nDecision x reason breakdown:")
    for decision, reasons in decision_counts.items():
        print(f"  {decision}:")
        for reason, count in reasons.items():
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
