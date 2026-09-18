"""
trainer_boundary.py — training loop cho BertCRFBoundaryNER (multi-task CRF +
boundary heads). Tách riêng khỏi trainer.py (dùng cho GuwenBertCRF/Linear)
vì optimizer param-group logic và per-epoch logging khác hẳn (differential
LR 4 nhóm, log riêng crf/start/end loss + P/R/F1 boundary head).
"""

import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.utils import EarlyStopping, CheckpointManager, get_logger, count_parameters

logger = get_logger(__name__)


class BoundaryTrainer:
    def __init__(self, model, cfg, output_dir: str, wandb_run=None):
        self.model      = model
        self.cfg        = cfg
        self.output_dir = output_dir
        self.wandb_run  = wandb_run
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        logger.info(f"Device: {self.device}")
        logger.info(f"Trainable params: {count_parameters(model):,}")

        tcfg = cfg.training
        precision = str(getattr(tcfg, "precision", "fp32") or "fp32").lower()
        self.use_fp16 = precision == "fp16" and self.device.type == "cuda"
        self.use_bf16 = precision == "bf16" and self.device.type == "cuda"
        self.scaler   = GradScaler(enabled=self.use_fp16)
        if self.use_fp16:
            logger.info("Mixed precision (fp16) enabled")
        elif self.use_bf16:
            logger.info("Mixed precision (bf16) enabled")

        self.ckpt_manager = CheckpointManager(
            output_dir, cfg.model.method, save_top_k=cfg.checkpoint.save_top_k
        )

        self.early_stop = None
        if getattr(cfg.early_stopping, "enabled", False):
            self.early_stop = EarlyStopping(
                patience=cfg.early_stopping.patience,
                min_delta=cfg.early_stopping.min_delta,
            )

        self.history: list = []

    # ── OPTIMIZER: 4 param group (mục E) ─────────────────────────────────────
    def _build_optimizer_scheduler(self, num_training_steps: int):
        tcfg = self.cfg.training
        no_decay = ("bias", "LayerNorm.weight")

        encoder_decay, encoder_no_decay = [], []
        for n, p in self.model.bert.named_parameters():
            (encoder_no_decay if any(nd in n for nd in no_decay) else encoder_decay).append(p)

        ner_head_params = (
            list(self.model.ner_classifier.parameters()) + list(self.model.crf.parameters())
        )
        boundary_head_params = (
            list(self.model.start_classifier.parameters())
            + list(self.model.end_classifier.parameters())
        )

        groups = [
            {"params": encoder_decay, "lr": tcfg.encoder_lr,
             "weight_decay": tcfg.weight_decay_encoder, "name": "encoder_decay"},
            {"params": encoder_no_decay, "lr": tcfg.encoder_lr,
             "weight_decay": 0.0, "name": "encoder_no_decay"},
            {"params": ner_head_params, "lr": tcfg.ner_head_lr,
             "weight_decay": tcfg.weight_decay_heads, "name": "ner_head_crf"},
            {"params": boundary_head_params, "lr": tcfg.boundary_head_lr,
             "weight_decay": tcfg.weight_decay_heads, "name": "boundary_heads"},
        ]
        # NOTE: crf_lr == ner_head_lr theo mặc định yêu cầu (cả 2 cùng 1e-4);
        # CRF transition params được gộp cùng nhóm ner_head_crf (dùng
        # tcfg.ner_head_lr) — nếu cần crf_lr khác ner_head_lr, tách nhóm
        # riêng bằng cfg.training.crf_lr (đã đọc nhưng mặc định bằng
        # ner_head_lr nên không tách nhóm thứ 5 để tránh phức tạp không cần
        # thiết khi giá trị luôn giống nhau theo yêu cầu mục E).
        optimizer = torch.optim.AdamW(groups)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * tcfg.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        self._group_names = [g["name"] for g in groups]
        return optimizer, scheduler

    def _current_lrs(self, optimizer) -> dict:
        return {name: g["lr"] for name, g in zip(self._group_names, optimizer.param_groups)}

    # ── TRAIN 1 EPOCH ──────────────────────────────────────────────────────
    def _train_epoch(self, loader, optimizer, scheduler) -> dict:
        self.model.train()
        sums = {"total": 0.0, "crf": 0.0, "start": 0.0, "end": 0.0}
        n_steps = 0

        for step, batch in enumerate(tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            ner_labels     = batch["labels"].to(self.device)
            start_labels   = batch.get("start_labels")
            end_labels     = batch.get("end_labels")
            if start_labels is not None:
                start_labels = start_labels.to(self.device)
            if end_labels is not None:
                end_labels = end_labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = autocast(dtype=torch.bfloat16) if self.use_bf16 else autocast(enabled=self.use_fp16)
            with autocast_ctx:
                out = self.model(
                    input_ids, attention_mask, token_type_ids,
                    ner_labels=ner_labels, start_labels=start_labels, end_labels=end_labels,
                )
                loss = out.loss

            if self.use_fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                optimizer.step()

            # Re-bake hard BIOES transition constraint sau MỖI optimizer step
            # (no-op nếu model.constrain_bioes_transitions=False) -- xem
            # src/bioes_utils.py:apply_hard_transition_constraints.
            if hasattr(self.model, "apply_transition_constraints"):
                self.model.apply_transition_constraints()

            scheduler.step()

            sums["total"] += float(loss.item())
            sums["crf"] += float(out.crf_loss.item()) if out.crf_loss is not None else 0.0
            sums["start"] += float(out.start_loss.item()) if out.start_loss is not None else 0.0
            sums["end"] += float(out.end_loss.item()) if out.end_loss is not None else 0.0
            n_steps += 1

        return {k: v / max(1, n_steps) for k, v in sums.items()}

    # ── MAIN LOOP ──────────────────────────────────────────────────────────
    def train(self, train_loader, dev_loader, evaluator) -> dict:
        tcfg = self.cfg.training
        total_steps = len(train_loader) * tcfg.epochs
        optimizer, scheduler = self._build_optimizer_scheduler(total_steps)

        best_dev_f1, best_epoch = 0.0, 0
        logger.info(f"Start training (boundary): {tcfg.epochs} epochs | {total_steps} steps")

        for epoch in range(1, tcfg.epochs + 1):
            t0 = time.time()
            logger.info(f"\n{'='*55}\nEpoch {epoch}/{tcfg.epochs}")

            train_losses = self._train_epoch(train_loader, optimizer, scheduler)
            dev_metrics = evaluator.evaluate(dev_loader)
            elapsed = time.time() - t0
            lrs = self._current_lrs(optimizer)

            per_class = {
                k: v for k, v in dev_metrics["report"].items()
                if isinstance(v, dict) and k not in ("micro avg", "macro avg", "weighted avg")
            }
            logger.info(
                f"  loss={train_losses['total']:.4f} (crf={train_losses['crf']:.4f} "
                f"start={train_losses['start']:.4f} end={train_losses['end']:.4f}) | "
                f"dev_f1={dev_metrics['f1']:.4f} | "
                f"dev_start_f1={dev_metrics['start_f1']:.4f} | "
                f"dev_end_f1={dev_metrics['end_f1']:.4f} | time={elapsed:.1f}s"
            )
            for ent, sc in per_class.items():
                logger.info(f"    {ent:8s} F1={sc['f1-score']:.4f} (support={sc['support']})")
            logger.info(f"  lr: {lrs}")

            if self.wandb_run:
                log_dict = {
                    "epoch": epoch,
                    "train_loss_total": train_losses["total"],
                    "train_loss_crf": train_losses["crf"],
                    "train_loss_start": train_losses["start"],
                    "train_loss_end": train_losses["end"],
                    "dev_f1": dev_metrics["f1"],
                    "dev_start_f1": dev_metrics["start_f1"],
                    "dev_end_f1": dev_metrics["end_f1"],
                }
                for ent, sc in per_class.items():
                    log_dict[f"dev_f1_{ent}"] = sc["f1-score"]
                for name, lr in lrs.items():
                    log_dict[f"lr_{name}"] = lr
                self.wandb_run.log(log_dict)

            # Checkpoint selection metric: strict entity-level micro-F1 trên dev
            # (mục E — KHÔNG chọn theo loss/token-accuracy/boundary-accuracy).
            is_best = self.ckpt_manager.save(self.model, dev_metrics["f1"], epoch)
            if is_best:
                best_dev_f1, best_epoch = dev_metrics["f1"], epoch
                logger.info(f"  New best! dev entity-F1={best_dev_f1:.4f}")

            self.history.append({
                "epoch": epoch,
                "train_loss_total": train_losses["total"],
                "train_loss_crf": train_losses["crf"],
                "train_loss_start": train_losses["start"],
                "train_loss_end": train_losses["end"],
                "dev_f1": dev_metrics["f1"],
                "dev_start_f1": dev_metrics["start_f1"],
                "dev_end_f1": dev_metrics["end_f1"],
                "lr": lrs,
                "elapsed": round(elapsed, 1),
            })

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if self.early_stop and self.early_stop.step(dev_metrics["f1"]):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        logger.info(f"\nBest: epoch={best_epoch} | dev_f1={best_dev_f1:.4f}")
        return {"best_epoch": best_epoch, "best_dev_f1": best_dev_f1, "history": self.history}
