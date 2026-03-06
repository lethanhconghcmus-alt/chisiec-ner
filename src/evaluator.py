"""
evaluator.py — F1 evaluation, error analysis, confusion matrix
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from TorchCRF import CRF
from collections import defaultdict
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOBES
 
from src.utils import get_logger, save_json

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, model, id2label: dict, device: torch.device, output_dir: str):
        self.model      = model
        self.id2label   = id2label
        self.device     = device
        self.output_dir = output_dir

    # ── CORE EVALUATE ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, dict]:
        self.model.eval()
        all_preds, all_labels, all_tokens_list = [], [], []

        for batch in tqdm(loader, desc="  eval ", leave=False, dynamic_ncols=True):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels         = batch["labels"]

            preds = self.model(input_ids, attention_mask, token_type_ids)

            for pred_seq, label_seq, mask in zip(preds, labels, attention_mask):
                true_tags, pred_tags = [], []
                for p, l, m in zip(pred_seq, label_seq.tolist(), mask.tolist()):
                    if m == 0 or l == -100:
                        continue
                    true_tags.append(self.id2label[l])
                    pred_tags.append(self.id2label[p])
                all_labels.append(true_tags)
                all_preds.append(pred_tags)

        report = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0,
            scheme=IOBES
        )
        f1 = f1_score(all_labels, all_preds, zero_division=0, scheme=IOBES)
        return f1, report

    # ── DETAILED REPORT ───────────────────────────────────────────────────────
    def full_report(self, loader, split: str = "test") -> dict:
        """Evaluate + lưu report chi tiết + sinh plots."""
        f1, report = self.evaluate(loader)

        logger.info(f"\n{'─'*55}")
        logger.info(f"[{split.upper()}] Overall F1 = {f1:.4f}")
        logger.info(f"{'─'*55}")
        for ent, sc in report.items():
            if isinstance(sc, dict):
                logger.info(
                    f"  {ent:25s} "
                    f"P={sc['precision']:.3f}  "
                    f"R={sc['recall']:.3f}  "
                    f"F1={sc['f1-score']:.3f}  "
                    f"(support={sc['support']})"
                )

        save_json(report, f"{self.output_dir}/{split}_report.json")
        return {"f1": f1, "report": report}

    # ── ERROR ANALYSIS ────────────────────────────────────────────────────────
    @torch.no_grad()
    def error_analysis(self, loader, raw_data: list, split: str = "test") -> dict:
        """
        Phân tích lỗi chi tiết:
        - False Positives / False Negatives theo entity type
        - Confusion examples (câu dự đoán sai)
        """
        self.model.eval()
        errors = defaultdict(list)       # {entity_type: [error_examples]}
        fp_by_type = defaultdict(int)    # false positives
        fn_by_type = defaultdict(int)    # false negatives

        batch_idx = 0
        for batch in tqdm(loader, desc="  error analysis", leave=False):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels         = batch["labels"]
            preds          = self.model(input_ids, attention_mask, token_type_ids)

            for b, (pred_seq, label_seq, mask) in enumerate(
                zip(preds, labels, attention_mask)
            ):
                true_tags, pred_tags = [], []
                for p, l, m in zip(pred_seq, label_seq.tolist(), mask.tolist()):
                    if m == 0 or l == -100:
                        continue
                    true_tags.append(self.id2label[l])
                    pred_tags.append(self.id2label[p])

                # Find mismatches
                for pos, (true, pred) in enumerate(zip(true_tags, pred_tags)):
                    if true != pred:
                        entity_type = true.split("-")[-1] if "-" in true else "O"
                        errors[entity_type].append({
                            "position": pos,
                            "true": true,
                            "pred": pred,
                        })
                        if true.startswith("B-") or true.startswith("I-"):
                            fn_by_type[entity_type] += 1
                        if pred.startswith("B-") or pred.startswith("I-"):
                            fp_by_type[pred.split("-")[-1]] += 1
                batch_idx += 1

        summary = {
            "false_positives_by_type": dict(fp_by_type),
            "false_negatives_by_type": dict(fn_by_type),
            "total_errors": sum(len(v) for v in errors.values()),
            "error_samples": {k: v[:5] for k, v in errors.items()},  # top 5 per type
        }
        save_json(summary, f"{self.output_dir}/{split}_error_analysis.json")
        logger.info(f"Error analysis saved → {self.output_dir}/{split}_error_analysis.json")

        # Plot FP/FN
        self._plot_fp_fn(fp_by_type, fn_by_type, split)
        return summary

    # ── CONFUSION MATRIX ──────────────────────────────────────────────────────
    @torch.no_grad()
    def confusion_matrix(self, loader, split: str = "test"):
        """Token-level confusion matrix cho các entity labels."""
        self.model.eval()
        label_list = sorted(self.id2label.values())
        label2idx  = {l: i for i, l in enumerate(label_list)}
        n = len(label_list)
        cm = np.zeros((n, n), dtype=int)

        for batch in tqdm(loader, desc="  confusion", leave=False):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels         = batch["labels"]
            preds          = self.model(input_ids, attention_mask, token_type_ids)

            for pred_seq, label_seq, mask in zip(preds, labels, attention_mask):
                for p, l, m in zip(pred_seq, label_seq.tolist(), mask.tolist()):
                    if m == 0 or l == -100:
                        continue
                    true_tag = self.id2label[l]
                    pred_tag = self.id2label[p]
                    cm[label2idx[true_tag], label2idx[pred_tag]] += 1

        # Plot
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_list, yticklabels=label_list,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {split}")
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
        plt.tight_layout()
        out = f"{self.output_dir}/{split}_confusion_matrix.png"
        plt.savefig(out, dpi=150); plt.close()
        logger.info(f"Confusion matrix saved → {out}")

    # ── PRIVATE PLOTS ─────────────────────────────────────────────────────────
    def _plot_fp_fn(self, fp: dict, fn: dict, split: str):
        if not fp and not fn:
            return
        entity_types = sorted(set(list(fp.keys()) + list(fn.keys())))
        x = range(len(entity_types))
        fp_vals = [fp.get(e, 0) for e in entity_types]
        fn_vals = [fn.get(e, 0) for e in entity_types]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - 0.2 for i in x], fp_vals, width=0.4, label="False Positives", color="#DD8452")
        ax.bar([i + 0.2 for i in x], fn_vals, width=0.4, label="False Negatives", color="#4C72B0")
        ax.set_xticks(list(x)); ax.set_xticklabels(entity_types, rotation=30, ha="right")
        ax.set_title(f"FP / FN by Entity Type — {split}")
        ax.legend(); plt.tight_layout()
        out = f"{self.output_dir}/{split}_fp_fn.png"
        plt.savefig(out, dpi=150); plt.close()
        logger.info(f"FP/FN chart saved → {out}")
