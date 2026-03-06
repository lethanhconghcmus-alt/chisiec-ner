"""
evaluate_compare.py — So sánh kết quả 3 phương pháp
Usage:
  python scripts/evaluate_compare.py --output_dir outputs
"""

import argparse
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METHODS = ["guwenbert_crf", "roberta_bilstm_crf", "roberta_kan_crf"]
LABELS  = ["GuwenBERT\n+CRF", "RoBERTa\n+BiLSTM+CRF", "RoBERTa\n+KAN+CRF"]
COLORS  = ["#4C72B0", "#DD8452", "#55A868"]


def load(output_dir, method):
    path = os.path.join(output_dir, method, "results.json")
    if not os.path.exists(path):
        print(f"⚠️  Not found: {path}")
        return None
    return json.load(open(path))


def summary_table(results):
    print("\n" + "="*70)
    print(f"{'Method':<25} {'Dev F1':>8} {'Test F1':>8} {'Best Ep':>8} {'Seed':>6}")
    print("="*70)
    rows = []
    for m, l, r in results:
        if r:
            print(f"{m:<25} {r['best_dev_f1']:>8.4f} {r['test_f1']:>8.4f} "
                  f"{r['best_epoch']:>8} {r.get('seed','-'):>6}")
            rows.append({"Method": m, "Dev F1": r["best_dev_f1"],
                         "Test F1": r["test_f1"], "Best Epoch": r["best_epoch"]})
    print("="*70)
    return rows


def entity_table(results):
    print("\n── Per-Entity Test F1 " + "─"*45)
    entity_set = set()
    for _, _, r in results:
        if r:
            for k, v in r["test_report"].items():
                if isinstance(v, dict):
                    entity_set.add(k)
    entity_list = sorted(entity_set)

    header = f"{'Entity':<20}" + "".join(f"  {l.replace(chr(10),' '):>22}" for _, l, r in results if r)
    print(header); print("─" * len(header))
    for ent in entity_list:
        row = f"{ent:<20}"
        for _, l, r in results:
            if r:
                sc  = r["test_report"].get(ent, {})
                f1  = sc.get("f1-score", 0) if isinstance(sc, dict) else 0
                row += f"  {f1:>22.4f}"
        print(row)


def plot_test_f1(results, output_dir):
    data = [(l, r["test_f1"]) for _, l, r in results if r]
    if not data: return
    labels, vals = zip(*data)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=COLORS[:len(labels)], width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim(0, min(1.05, max(vals) * 1.15))
    ax.set_title("Test F1 Comparison — CHisIEC NER", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "compare_test_f1.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n📊 Saved → {out}")


def plot_training_curves(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (_, label, r), color in zip(results, COLORS):
        if not r: continue
        label_clean = label.replace("\n", " ")
        epochs = [h["epoch"]    for h in r["history"]]
        losses = [h["train_loss"] for h in r["history"]]
        f1s    = [h["dev_f1"]   for h in r["history"]]
        axes[0].plot(epochs, losses, marker="o", label=label_clean, color=color)
        axes[1].plot(epochs, f1s,    marker="o", label=label_clean, color=color)

    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].set_title("Dev F1 Score");  axes[1].set_xlabel("Epoch"); axes[1].legend()
    for ax in axes: ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "compare_training_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"📊 Saved → {out}")


def plot_per_entity_f1(results, output_dir):
    """Grouped bar chart: entity types × methods."""
    entity_set = set()
    for _, _, r in results:
        if r:
            for k, v in r["test_report"].items():
                if isinstance(v, dict) and k not in ("micro avg","macro avg","weighted avg"):
                    entity_set.add(k)
    entities = sorted(entity_set)
    if not entities: return

    n = len(entities)
    x = range(n)
    valid = [(l, r) for _, l, r in results if r]
    width = 0.8 / len(valid)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))
    for i, (label, r) in enumerate(valid):
        f1s = [r["test_report"].get(e, {}).get("f1-score", 0)
               if isinstance(r["test_report"].get(e, {}), dict) else 0
               for e in entities]
        offset = [xi + (i - len(valid)/2 + 0.5) * width for xi in x]
        ax.bar(offset, f1s, width=width, label=label.replace("\n"," "),
               color=COLORS[i], edgecolor="white")

    ax.set_xticks(list(x)); ax.set_xticklabels(entities, rotation=30, ha="right")
    ax.set_title("Per-Entity F1 — Test Set"); ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "compare_entity_f1.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"📊 Saved → {out}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    results = [(m, l, load(args.output_dir, m)) for m, l in zip(METHODS, LABELS)]

    rows = summary_table(results)
    entity_table(results)
    plot_test_f1(results, args.output_dir)
    plot_training_curves(results, args.output_dir)
    plot_per_entity_f1(results, args.output_dir)

    # CSV
    if rows:
        df = pd.DataFrame(rows)
        csv = os.path.join(args.output_dir, "comparison_summary.csv")
        df.to_csv(csv, index=False)
        print(f"📄 CSV → {csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs")
    main(p.parse_args())
