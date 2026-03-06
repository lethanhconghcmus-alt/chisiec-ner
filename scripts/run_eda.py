"""
run_eda.py — EDA + Data Validation cho CHisIEC NER dataset
Usage:
  python scripts/run_eda.py --data_dir data --output_dir outputs/eda
"""

import argparse
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data_utils import read_conll, validate_data
from src.utils import get_logger, save_json

logger = get_logger(__name__)


def compute_stats(data: list, split: str) -> dict:
    token_lengths = [len(t) for t, _ in data]
    all_labels    = [l for _, labs in data for l in labs]
    label_counter = Counter(all_labels)
    entity_counter = Counter({
        k: v for k, v in label_counter.items() if k != "O"
    })

    return {
        "split":             split,
        "num_sentences":     len(data),
        "total_tokens":      sum(token_lengths),
        "avg_len":           round(sum(token_lengths) / len(token_lengths), 2),
        "max_len":           max(token_lengths),
        "min_len":           min(token_lengths),
        "label_distribution": dict(label_counter.most_common()),
        "entity_distribution": dict(entity_counter.most_common()),
        "o_ratio":           round(label_counter["O"] / max(len(all_labels), 1), 3),
    }


def plot_length_distribution(splits_data: dict, output_dir: str):
    fig, axes = plt.subplots(1, len(splits_data), figsize=(5 * len(splits_data), 4), sharey=True)
    if len(splits_data) == 1:
        axes = [axes]
    for ax, (split, data) in zip(axes, splits_data.items()):
        lengths = [len(t) for t, _ in data]
        ax.hist(lengths, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(f"{split} sentence lengths")
        ax.set_xlabel("# tokens"); ax.set_ylabel("count")
        ax.axvline(sum(lengths)/len(lengths), color="red", linestyle="--", label=f"mean={sum(lengths)/len(lengths):.1f}")
        ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, "sentence_lengths.png")
    plt.savefig(out, dpi=150); plt.close()
    logger.info(f"Saved → {out}")


def plot_label_distribution(splits_stats: list, output_dir: str):
    for stat in splits_stats:
        split  = stat["split"]
        labels = stat["label_distribution"]
        df     = pd.DataFrame(list(labels.items()), columns=["Label", "Count"])
        df     = df.sort_values("Count", ascending=False)

        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))
        colors  = ["#DD8452" if l != "O" else "#4C72B0" for l in df["Label"]]
        ax.bar(df["Label"], df["Count"], color=colors, edgecolor="white")
        ax.set_title(f"Label distribution — {split}")
        ax.set_xlabel("Label"); ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        out = os.path.join(output_dir, f"{split}_label_dist.png")
        plt.savefig(out, dpi=150); plt.close()
        logger.info(f"Saved → {out}")


def plot_entity_heatmap(splits_stats: list, output_dir: str):
    """Heatmap entity counts: rows=entity type, cols=split"""
    entity_set = set()
    for st in splits_stats:
        entity_set.update(st["entity_distribution"].keys())
    entity_list = sorted(entity_set)

    matrix = []
    splits  = []
    for st in splits_stats:
        splits.append(st["split"])
        matrix.append([st["entity_distribution"].get(e, 0) for e in entity_list])

    import numpy as np
    df = pd.DataFrame(np.array(matrix).T, index=entity_list, columns=splits)

    fig, ax = plt.subplots(figsize=(max(6, len(splits)*2), max(4, len(entity_list)*0.5)))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Entity counts per split")
    plt.tight_layout()
    out = os.path.join(output_dir, "entity_heatmap.png")
    plt.savefig(out, dpi=150); plt.close()
    logger.info(f"Saved → {out}")


def print_summary_table(splits_stats: list):
    print("\n" + "="*65)
    print(f"{'Split':<10} {'Sents':>8} {'Tokens':>8} {'AvgLen':>8} {'O-ratio':>8}")
    print("="*65)
    for st in splits_stats:
        print(
            f"{st['split']:<10} {st['num_sentences']:>8,} "
            f"{st['total_tokens']:>8,} {st['avg_len']:>8.1f} "
            f"{st['o_ratio']:>8.3f}"
        )
    print("="*65)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    splits_data  = {}
    splits_stats = []

    for split in ["train", "dev", "test"]:
        path = os.path.join(args.data_dir, f"{split}.txt")
        if not os.path.exists(path):
            logger.warning(f"Not found: {path}, skipping")
            continue
        data = read_conll(path)
        splits_data[split] = data

        # Validate
        val_stats = validate_data(data, split)
        stats     = compute_stats(data, split)
        stats.update({"bio_errors": val_stats["bio_errors"], "errors": val_stats["errors"]})
        splits_stats.append(stats)
        save_json(stats, os.path.join(args.output_dir, f"{split}_stats.json"))

    print_summary_table(splits_stats)
    plot_length_distribution(splits_data, args.output_dir)
    plot_label_distribution(splits_stats, args.output_dir)
    plot_entity_heatmap(splits_stats, args.output_dir)

    save_json(splits_stats, os.path.join(args.output_dir, "all_stats.json"))
    logger.info(f"\n✅ EDA complete. Output → {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--output_dir", default="outputs/eda")
    main(p.parse_args())
