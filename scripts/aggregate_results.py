"""
aggregate_results.py — tổng hợp kết quả multi-seed cho các experiment
M0/M1/M2 (mục G). Đọc nhiều file results.json (ghi bởi scripts/train.py
hoặc scripts/train_boundary.py, mỗi cái ứng với 1 (experiment, seed)),
gộp thành 1 bảng CSV/Markdown: mean/std/min/max theo experiment, và
delta M2-M1 / M1-M0 theo từng seed (nếu pairing được).

Usage:
  python scripts/aggregate_results.py \
      --exp M0="outputs/ancient_m0/*/results.json" \
      --exp M1="outputs/ancient_m1/*/results.json" \
      --exp M2="outputs/ancient_boundary_m2/*/results.json" \
      --out results/ablation_summary

Mỗi results.json PHẢI có "seed" (script train tự ghi). Nếu 1 experiment
chạy nhiều seed vào chung 1 checkpoint.output_dir (bị ghi đè), hãy đặt
output_dir riêng theo seed (vd checkpoint.output_dir=outputs/ancient_m0/seed42)
khi gọi scripts/train.py — script này KHÔNG tự suy seed từ path.
"""

import argparse
import csv
import glob
import json
import statistics as stats
from collections import defaultdict

METRIC_KEYS = [
    "best_epoch", "best_dev_f1", "test_f1", "test_precision", "test_recall",
    "test_PER_f1", "test_LOC_f1", "test_ORG_f1", "test_TITLE_f1", "test_DTM_f1",
    "start_f1", "end_f1", "total_params", "train_time",
]


def _get_class_f1(report: dict, cls: str):
    sc = report.get(cls)
    return sc["f1-score"] if isinstance(sc, dict) else None


def _load_one(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    report = data.get("test_report", {})
    micro = report.get("micro avg", {})
    row = {
        "seed": data.get("seed"),
        "best_epoch": data.get("best_epoch"),
        "best_dev_f1": data.get("best_dev_f1"),
        "test_f1": data.get("test_f1"),
        "test_precision": micro.get("precision"),
        "test_recall": micro.get("recall"),
        "test_PER_f1": _get_class_f1(report, "PER"),
        "test_LOC_f1": _get_class_f1(report, "LOC"),
        "test_ORG_f1": _get_class_f1(report, "ORG"),
        "test_TITLE_f1": _get_class_f1(report, "TITLE"),
        "test_DTM_f1": _get_class_f1(report, "DTM"),
        "start_f1": data.get("test_start_f1"),
        "end_f1": data.get("test_end_f1"),
        "total_params": data.get("total_params"),
        "train_time": data.get("train_time"),
        "_path": path,
    }
    return row


def load_experiment(name: str, pattern: str) -> list:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"[{name}] Không tìm thấy file nào khớp pattern: {pattern}")
    rows = []
    seen_seeds = set()
    for p in paths:
        row = _load_one(p)
        row["experiment"] = name
        if row["seed"] in seen_seeds:
            raise ValueError(
                f"[{name}] Trùng seed={row['seed']} giữa nhiều file (vd {p}) — "
                f"mỗi seed phải ứng với đúng 1 checkpoint.output_dir riêng."
            )
        seen_seeds.add(row["seed"])
        rows.append(row)
    return rows


def summarize(rows: list) -> dict:
    summary = {"experiment": rows[0]["experiment"], "n_seeds": len(rows),
               "seeds": [r["seed"] for r in rows]}
    for key in METRIC_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            summary[f"{key}_mean"] = summary[f"{key}_std"] = None
            summary[f"{key}_min"] = summary[f"{key}_max"] = None
            continue
        summary[f"{key}_mean"] = round(stats.fmean(vals), 4)
        summary[f"{key}_std"] = round(stats.pstdev(vals), 4) if len(vals) > 1 else 0.0
        summary[f"{key}_min"] = round(min(vals), 4)
        summary[f"{key}_max"] = round(max(vals), 4)
    return summary


def paired_deltas(rows_a: list, rows_b: list, metric: str = "test_f1") -> dict:
    """rows_b - rows_a theo từng seed chung. Trả {seed: delta}."""
    by_seed_a = {r["seed"]: r.get(metric) for r in rows_a}
    by_seed_b = {r["seed"]: r.get(metric) for r in rows_b}
    common = sorted(set(by_seed_a) & set(by_seed_b), key=lambda s: (s is None, s))
    deltas = {}
    for seed in common:
        va, vb = by_seed_a[seed], by_seed_b[seed]
        if va is None or vb is None:
            continue
        deltas[seed] = round(vb - va, 4)
    return deltas


def write_csv(all_rows: list, path: str):
    fieldnames = ["experiment", "seed"] + METRIC_KEYS + ["_path"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_markdown(summaries: list, deltas: dict, path: str):
    lines = ["| experiment | n_seeds | test_f1 (mean±std) | PER | LOC | ORG | TITLE | DTM | start_f1 | end_f1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        def fmt(k):
            m, sd = s.get(f"{k}_mean"), s.get(f"{k}_std")
            return f"{m:.4f}±{sd:.4f}" if m is not None else "-"
        lines.append(
            f"| {s['experiment']} | {s['n_seeds']} | {fmt('test_f1')} | {fmt('test_PER_f1')} | "
            f"{fmt('test_LOC_f1')} | {fmt('test_ORG_f1')} | {fmt('test_TITLE_f1')} | "
            f"{fmt('test_DTM_f1')} | {fmt('start_f1')} | {fmt('end_f1')} |"
        )
    lines.append("")
    lines.append("### Delta theo từng seed (test_f1)")
    for pair_name, d in deltas.items():
        lines.append(f"\n**{pair_name}**")
        lines.append("| seed | delta |")
        lines.append("|---|---|")
        for seed, val in d.items():
            lines.append(f"| {seed} | {val:+.4f} |")
        if d:
            mean_delta = round(stats.fmean(d.values()), 4)
            lines.append(f"| **mean** | **{mean_delta:+.4f}** |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", required=True,
                     help='NAME=glob_pattern, có thể lặp lại (vd --exp M0="out/m0/*/results.json")')
    ap.add_argument("--out", default="results/ablation_summary",
                     help="prefix output (ghi <out>.csv và <out>.md)")
    args = ap.parse_args()

    experiments = {}
    for item in args.exp:
        name, pattern = item.split("=", 1)
        experiments[name] = load_experiment(name, pattern)

    all_rows = [r for rows in experiments.values() for r in rows]
    summaries = [summarize(rows) for rows in experiments.values()]

    deltas = {}
    if "M1" in experiments and "M0" in experiments:
        deltas["M1 - M0"] = paired_deltas(experiments["M0"], experiments["M1"])
    if "M2" in experiments and "M1" in experiments:
        deltas["M2 - M1"] = paired_deltas(experiments["M1"], experiments["M2"])

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_csv(all_rows, f"{args.out}.csv")
    write_markdown(summaries, deltas, f"{args.out}.md")

    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "deltas": deltas}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out}.csv, {args.out}.md, {args.out}.json")
    for s in summaries:
        print(f"  {s['experiment']}: n={s['n_seeds']} test_f1={s.get('test_f1_mean')}±{s.get('test_f1_std')}")


if __name__ == "__main__":
    main()
