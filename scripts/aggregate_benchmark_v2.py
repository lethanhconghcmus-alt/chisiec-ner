"""
aggregate_benchmark_v2.py — tổng hợp kết quả benchmark data-centric
R0 (dataset_v1) / R1 (dataset_v2_full) / R2 (dataset_v2_clean), 3 seed
mỗi experiment (42/43/44), SikuBERT+Linear+CRF+BIO cố định (xem
configs/benchmark_v2/R{0,1,2}_*.yaml).

Đọc results.json (ghi bởi scripts/train.py, đã có dataset_manifest +
extended_report_summary từ src/evaluator_extended.py) + file đứng cạnh
test_extended_report.json (đầy đủ hơn, có gr11_dtm_examples/
confusable_pair_counts mà results.json không nhúng full).

Usage:
  python scripts/aggregate_benchmark_v2.py \
      --run R0="outputs/benchmark_v2/R0_v1/seed*/guwenbert_crf" \
      --run R1="outputs/benchmark_v2/R1_v2_full/seed*/guwenbert_crf" \
      --run R2="outputs/benchmark_v2/R2_v2_clean/seed*/guwenbert_crf" \
      --out-dir artifacts/benchmark_v2

KHÔNG tự claim significance -- chỉ báo delta thô theo seed + mean/std,
không chạy paired t-test/bootstrap trừ khi được yêu cầu riêng.
"""

import argparse
import csv
import glob
import json
import os
import statistics as stats
from collections import defaultdict

ENTITY_TYPES = ["PER", "LOC", "ORG", "TITLE", "DTM"]

METRIC_KEYS = [
    "best_epoch", "best_dev_f1", "test_f1", "test_precision", "test_recall",
    "n_sentences_train", "n_entities_train", "n_sentences_dev", "n_entities_dev",
    "n_sentences_test", "n_entities_test",
    "seen_f1", "unseen_f1",
    "raw_bio_violations_total", "raw_sentences_with_violations",
    "constrained_bio_violations_total", "constrained_sentences_with_violations",
    "org_gold_pred_title", "title_gold_pred_org",
    "total_params", "train_time",
]


def _get(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load_run(path: str) -> dict:
    results_path = os.path.join(path, "results.json")
    with open(results_path, encoding="utf-8") as f:
        r = json.load(f)

    ext_path = os.path.join(path, "test_extended_report.json")
    ext = {}
    if os.path.exists(ext_path):
        with open(ext_path, encoding="utf-8") as f:
            ext = json.load(f)

    report = r.get("test_report", {})
    micro = report.get("micro avg", {})
    manifest = r.get("dataset_manifest", {})
    ext_summary = r.get("extended_report_summary") or {}
    bio_v = ext_summary.get("bio_violations") or ext.get("bio_violations") or {}
    seen_unseen = ext_summary.get("seen_unseen_f1") or ext.get("seen_unseen_f1") or {}
    org_title = ext_summary.get("org_title_confusion") or ext.get("org_title_confusion") or {}
    span_cats = ext.get("span_category_counts") or (ext_summary.get("span_category_counts")
                                                      if "span_category_counts" in ext_summary else {})

    row = {
        "experiment": None, "seed": r.get("seed"), "run_path": path,
        "best_epoch": r.get("best_epoch"), "best_dev_f1": r.get("best_dev_f1"),
        "test_f1": r.get("test_f1"),
        "test_precision": micro.get("precision"), "test_recall": micro.get("recall"),
        "n_sentences_train": manifest.get("n_sentences_train"),
        "n_entities_train": manifest.get("n_entities_train"),
        "n_sentences_dev": manifest.get("n_sentences_dev"),
        "n_entities_dev": manifest.get("n_entities_dev"),
        "n_sentences_test": manifest.get("n_sentences_test"),
        "n_entities_test": manifest.get("n_entities_test"),
        "dataset_checksums": manifest.get("checksums_md5"),
        "decode_policy": ext.get("decode_policy") or (ext_summary.get("decode_policy")
                                                        if "decode_policy" in ext_summary else None),
        "seen_f1": _get(seen_unseen, "seen", "f1"), "unseen_f1": _get(seen_unseen, "unseen", "f1"),
        "raw_bio_violations_total": bio_v.get("raw_total_violations"),
        "raw_sentences_with_violations": bio_v.get("raw_sentences_with_violations"),
        "constrained_bio_violations_total": bio_v.get("constrained_total_violations"),
        "constrained_sentences_with_violations": bio_v.get("constrained_sentences_with_violations"),
        "org_gold_pred_title": org_title.get("gold_ORG_pred_TITLE"),
        "title_gold_pred_org": org_title.get("gold_TITLE_pred_ORG"),
        "total_params": r.get("total_params"), "train_time": r.get("train_time"),
        "_report": report, "_span_category_counts": span_cats,
        "_gr11_examples": ext.get("gr11_dtm_examples", []),
    }
    for etype in ENTITY_TYPES:
        sc = report.get(etype)
        row[f"{etype}_precision"] = sc["precision"] if isinstance(sc, dict) else None
        row[f"{etype}_recall"] = sc["recall"] if isinstance(sc, dict) else None
        row[f"{etype}_f1"] = sc["f1-score"] if isinstance(sc, dict) else None
        row[f"{etype}_support"] = sc["support"] if isinstance(sc, dict) else None
    return row


def load_experiment(name: str, pattern: str, expected_seeds: set) -> list:
    paths = sorted(p for p in glob.glob(pattern) if os.path.isdir(p))
    if not paths:
        raise FileNotFoundError(f"[{name}] Khong tim thay run nao khop pattern: {pattern}")
    rows = []
    seen_seeds = set()
    for p in paths:
        row = load_run(p)
        row["experiment"] = name
        if row["seed"] in seen_seeds:
            raise ValueError(f"[{name}] Trung seed={row['seed']} giua nhieu run ({p}).")
        seen_seeds.add(row["seed"])
        rows.append(row)

    # ── FAIL RO neu thieu seed (khong duoc am tham chay voi it hon 3 seed) ──
    missing = expected_seeds - seen_seeds
    if missing:
        raise ValueError(
            f"[{name}] THIEU output cho seed {sorted(missing)} (co seed {sorted(seen_seeds)}, "
            f"can du {sorted(expected_seeds)}). Kiem tra lai job da chay xong chua truoc khi aggregate."
        )
    extra = seen_seeds - expected_seeds
    if extra:
        raise ValueError(
            f"[{name}] Co seed KHONG mong doi {sorted(extra)} (--expected-seeds={sorted(expected_seeds)}). "
            f"Neu day la co y (them seed), truyen --expected-seeds day du."
        )

    # ── FAIL RO neu dataset manifest (checksum) LECH giua cac seed cung
    # experiment -- cac seed trong 1 experiment PHAI dung CHINH XAC 1 bo
    # data giong het nhau, chi khac random seed. Lech checksum nghia la co
    # seed nao do da chay tren data khac (vd materialize lai giua chung,
    # hoac tro nham path). ──
    checksums_by_seed = {r["seed"]: r.get("dataset_checksums") for r in rows}
    distinct = {tuple(sorted((c or {}).items())) for c in checksums_by_seed.values()}
    if len(distinct) > 1:
        raise ValueError(
            f"[{name}] DATASET MANIFEST CHECKSUM LECH giua cac seed cung experiment: "
            f"{checksums_by_seed} -- cac seed PHAI chay tren cung 1 bo file du lieu. "
            f"KHONG aggregate khi con lech nay."
        )
    if not any(checksums_by_seed.values()):
        raise ValueError(
            f"[{name}] KHONG doc duoc dataset_checksums tu bat ky run nao (results.json thieu "
            f"dataset_manifest) -- co the dang doc ket qua tu ban train.py CU chua co truong nay."
        )
    return rows


def summarize(rows: list) -> dict:
    keys = METRIC_KEYS + [f"{e}_{m}" for e in ENTITY_TYPES for m in ("precision", "recall", "f1", "support")]
    summary = {"experiment": rows[0]["experiment"], "n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            summary[f"{key}_mean"] = summary[f"{key}_std"] = summary[f"{key}_min"] = summary[f"{key}_max"] = None
            continue
        summary[f"{key}_mean"] = round(stats.fmean(vals), 4)
        summary[f"{key}_std"] = round(stats.pstdev(vals), 4) if len(vals) > 1 else 0.0
        summary[f"{key}_min"] = round(min(vals), 4)
        summary[f"{key}_max"] = round(max(vals), 4)
    return summary


def paired_deltas(rows_a: list, rows_b: list, metric: str = "test_f1") -> dict:
    """rows_b - rows_a theo tung seed CHUNG. KHONG claim significance."""
    by_a = {r["seed"]: r.get(metric) for r in rows_a}
    by_b = {r["seed"]: r.get(metric) for r in rows_b}
    common = sorted(set(by_a) & set(by_b), key=lambda s: (s is None, s))
    return {seed: round(by_b[seed] - by_a[seed], 4) for seed in common
            if by_a[seed] is not None and by_b[seed] is not None}


def write_per_seed_csv(all_rows: list, path: str):
    fieldnames = ["experiment", "seed", "run_path"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def write_aggregate_csv(summaries: list, path: str):
    keys = METRIC_KEYS + [f"{e}_{m}" for e in ENTITY_TYPES for m in ("precision", "recall", "f1", "support")]
    fieldnames = ["experiment", "n_seeds"] + [f"{k}_{stat}" for k in keys for stat in ("mean", "std", "min", "max")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in fieldnames})


def write_per_label_csv(all_rows: list, path: str):
    fieldnames = ["experiment", "seed"] + [f"{e}_{m}" for e in ENTITY_TYPES for m in ("precision", "recall", "f1", "support")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def write_paired_deltas_csv(deltas: dict, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "seed", "delta_test_f1"])
        for name, d in deltas.items():
            for seed, val in d.items():
                w.writerow([name, seed, val])
            if d:
                w.writerow([name, "mean", round(stats.fmean(d.values()), 4)])


def write_dataset_statistics_csv(all_rows: list, path: str):
    fieldnames = ["experiment", "seed", "n_sentences_train", "n_entities_train",
                  "n_sentences_dev", "n_entities_dev", "n_sentences_test", "n_entities_test",
                  "dataset_checksums"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def write_benchmark_summary_md(summaries: list, deltas: dict, path: str):
    lines = [
        "# Benchmark v2 data-centric — R0 (v1) / R1 (v2_full) / R2 (v2_clean)",
        "",
        "Kien truc co dinh: SikuBERT + Linear + CRF + BIO. KHONG BIOES/boundary "
        "heads/DAPT/gazetteer/GlobalPointer/LLM/augmentation.",
        "",
        "**R1/R2 la evaluation set KHAC v1** (v2_full giu nguyen so cau nhung sua "
        "nhan; v2_clean loai 151 sample quarantine -> train/dev/test composition "
        "khac v1). KHONG so sanh test F1 truc tiep nhu the cung 1 test set.",
        "",
        "## Aggregate (mean ± std qua 3 seed 42/43/44)",
        "Test F1 dùng CONSTRAINED decode tags khi `evaluation.constrained_decode=true` "
        "(xem artifacts/benchmark_v2/bio_decoding_policy.md) -- `constrained_bio_violations` "
        "phải = 0 nếu policy được áp dụng đúng.",
        "",
        "| experiment | n_seeds | test_f1 | PER | LOC | ORG | TITLE | DTM | seen_f1 | unseen_f1 | raw_bio_viol | constrained_bio_viol |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        def fmt(k):
            m, sd = s.get(f"{k}_mean"), s.get(f"{k}_std")
            return f"{m:.4f}±{sd:.4f}" if m is not None else "-"
        lines.append(
            f"| {s['experiment']} | {s['n_seeds']} | {fmt('test_f1')} | {fmt('PER_f1')} | "
            f"{fmt('LOC_f1')} | {fmt('ORG_f1')} | {fmt('TITLE_f1')} | {fmt('DTM_f1')} | "
            f"{fmt('seen_f1')} | {fmt('unseen_f1')} | {fmt('raw_bio_violations_total')} | "
            f"{fmt('constrained_bio_violations_total')} |"
        )
    lines += ["", "## Paired delta theo seed (test_f1, KHONG claim significance)"]
    for name, d in deltas.items():
        lines.append(f"\n**{name}**")
        lines.append("| seed | delta |")
        lines.append("|---|---|")
        for seed, val in d.items():
            lines.append(f"| {seed} | {val:+.4f} |")
        if d:
            lines.append(f"| **mean** | **{round(stats.fmean(d.values()), 4):+.4f}** |")
    lines.append("")
    lines.append(
        "*Lưu ý: chỉ là delta thô (raw), không phải kiểm định thống kê. "
        "Không suy diễn 'significant' từ bảng này nếu chưa chạy paired test rõ ràng.*"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_error_analysis_summary_md(all_rows: list, path: str):
    lines = ["# Error analysis summary — benchmark v2", ""]
    by_exp = defaultdict(list)
    for r in all_rows:
        by_exp[r["experiment"]].append(r)
    for exp, rows in by_exp.items():
        lines.append(f"## {exp}")
        cat_totals = defaultdict(int)
        for r in rows:
            for cat, cnt in (r.get("_span_category_counts") or {}).items():
                cat_totals[cat] += cnt
        if cat_totals:
            lines.append("Span category counts (tổng qua các seed, đơn vị = số câu):")
            for cat, cnt in sorted(cat_totals.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {cat}: {cnt}")
        org_title = sum((r.get("org_gold_pred_title") or 0) for r in rows)
        title_org = sum((r.get("title_gold_pred_org") or 0) for r in rows)
        lines.append(f"- ORG(gold)->TITLE(pred): {org_title} | TITLE(gold)->ORG(pred): {title_org}")
        n_gr11 = sum(len(r.get("_gr11_examples") or []) for r in rows)
        lines.append(f"- GR-11 DTM example count sampled: {n_gr11} (xem test_extended_report.json từng seed để xem chi tiết)")
        lines.append(f"- Raw (unconstrained) BIO violations (tổng qua seed): "
                      f"{sum((r.get('raw_bio_violations_total') or 0) for r in rows)}")
        lines.append(f"- Constrained BIO violations (tổng qua seed, PHẢI = 0 nếu decode_policy=constrained): "
                      f"{sum((r.get('constrained_bio_violations_total') or 0) for r in rows)}")
        decode_policies = {r.get("decode_policy") for r in rows if r.get("decode_policy")}
        lines.append(f"- decode_policy dùng: {sorted(decode_policies) if decode_policies else 'khong ro (results.json cu?)'}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                     help='NAME=glob_pattern (thư mục chứa results.json), lặp lại cho R0/R1/R2. '
                          'Ví dụ: --run R0="outputs/benchmark_v2/R0_v1/seed*/guwenbert_crf"')
    ap.add_argument("--out-dir", default="artifacts/benchmark_v2")
    ap.add_argument("--expected-seeds", default="42,43,44",
                     help="Danh sach seed BAT BUOC phai co du cho MOI experiment (mac dinh 42,43,44). "
                          "Aggregate se FAIL ro neu thieu hoac thua seed so voi danh sach nay.")
    args = ap.parse_args()

    expected_seeds = {int(s) for s in args.expected_seeds.split(",") if s.strip()}

    experiments = {}
    for item in args.run:
        name, pattern = item.split("=", 1)
        experiments[name] = load_experiment(name, pattern, expected_seeds)

    all_rows = [r for rows in experiments.values() for r in rows]
    summaries = [summarize(rows) for rows in experiments.values()]

    deltas = {}
    if "R0" in experiments and "R1" in experiments:
        deltas["R1 - R0"] = paired_deltas(experiments["R0"], experiments["R1"])
    if "R0" in experiments and "R2" in experiments:
        deltas["R2 - R0"] = paired_deltas(experiments["R0"], experiments["R2"])
    if "R1" in experiments and "R2" in experiments:
        deltas["R2 - R1"] = paired_deltas(experiments["R1"], experiments["R2"])

    os.makedirs(args.out_dir, exist_ok=True)
    write_per_seed_csv(all_rows, os.path.join(args.out_dir, "per_seed_results.csv"))
    write_aggregate_csv(summaries, os.path.join(args.out_dir, "aggregate_results.csv"))
    write_paired_deltas_csv(deltas, os.path.join(args.out_dir, "paired_deltas.csv"))
    write_per_label_csv(all_rows, os.path.join(args.out_dir, "per_label_results.csv"))
    write_dataset_statistics_csv(all_rows, os.path.join(args.out_dir, "dataset_statistics.csv"))
    write_benchmark_summary_md(summaries, deltas, os.path.join(args.out_dir, "benchmark_summary.md"))
    write_error_analysis_summary_md(all_rows, os.path.join(args.out_dir, "error_analysis_summary.md"))

    print(f"Wrote 7 files -> {args.out_dir}/")
    for s in summaries:
        print(f"  {s['experiment']}: n={s['n_seeds']} test_f1={s.get('test_f1_mean')}±{s.get('test_f1_std')}")


if __name__ == "__main__":
    main()
