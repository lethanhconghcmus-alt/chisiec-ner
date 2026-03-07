"""
trainer.py — Training loop với fp16, WandB logging, early stopping
"""

import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.utils import (
    EarlyStopping, CheckpointManager, get_logger, save_json, count_parameters
)

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, cfg, output_dir: str, wandb_run=None):
        self.model      = model
        self.cfg        = cfg
        self.output_dir = output_dir
        self.wandb_run  = wandb_run
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        logger.info(f"Device: {self.device}")
        logger.info(f"Trainable params: {count_parameters(model):,}")

        # fp16 scaler
        self.use_fp16 = cfg.training.fp16 and self.device.type == "cuda"
        self.scaler   = GradScaler() if self.use_fp16 else None
        if self.use_fp16:
            logger.info("Mixed precision (fp16) enabled")

        self.ckpt_manager = CheckpointManager(
            output_dir, cfg.model.method, save_top_k=cfg.checkpoint.save_top_k
        )
        self.early_stop   = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
        ) if cfg.early_stopping.enabled else None

        self.history: list[dict] = []

    # ── OPTIMIZER & SCHEDULER ─────────────────────────────────────────────────
    def _build_optimizer_scheduler(self, num_training_steps: int):
        tcfg = self.cfg.training
        bert_params  = list(self.model.bert.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if not n.startswith("bert")
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params,  "lr": tcfg.bert_lr},
                {"params": other_params, "lr": tcfg.head_lr},
            ],
            weight_decay=tcfg.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * tcfg.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    # ── TRAIN ONE EPOCH ───────────────────────────────────────────────────────
    def _train_epoch(self, loader, optimizer, scheduler) -> float:
        self.model.train()
        total_loss, n_steps = 0.0, 0

        for step, batch in enumerate(tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)):
            if step % 50 == 0:
                print(f"[train] step={step}", flush=True)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                 token_type_ids = token_type_ids.to(self.device)
            labels         = batch["labels"].to(self.device)

            optimizer.zero_grad()

            if self.use_fp16:
                with autocast():
                    emissions = emissions.float()
                    loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss, _ = self.model(input_ids, attention_mask, token_type_ids, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            n_steps    += 1

        return total_loss / n_steps

    # ── MAIN TRAIN LOOP ───────────────────────────────────────────────────────
    def train(self, train_loader, dev_loader, evaluator) -> dict:
        tcfg          = self.cfg.training
        total_steps   = len(train_loader) * tcfg.epochs
        optimizer, scheduler = self._build_optimizer_scheduler(total_steps)

        best_dev_f1, best_epoch = 0.0, 0
        logger.info(f"Start training: {tcfg.epochs} epochs | {total_steps} steps")

        for epoch in range(1, tcfg.epochs + 1):
            t0 = time.time()
            logger.info(f"\n{'='*55}\nEpoch {epoch}/{tcfg.epochs}")
            print(f"\n>>> Epoch {epoch}/{tcfg.epochs} bắt đầu...", flush=True)
            train_loss      = self._train_epoch(train_loader, optimizer, scheduler)
            dev_f1, dev_rep = evaluator.evaluate(dev_loader)
            elapsed         = time.time() - t0

            logger.info(
                f"  loss={train_loss:.4f} | dev_f1={dev_f1:.4f} | "
                f"time={elapsed:.1f}s"
            )
            print(f"[Epoch {epoch}/{tcfg.epochs}] loss={train_loss:.4f} | dev_f1={dev_f1:.4f} | time={elapsed:.1f}s", flush=True)  # ← thêm

            # WandB log
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": epoch, "train_loss": train_loss,
                    "dev_f1": dev_f1,
                })

            # Checkpoint
            is_best = self.ckpt_manager.save(self.model, dev_f1, epoch)
            if is_best:
                best_dev_f1, best_epoch = dev_f1, epoch
                logger.info(f"  ✅ New best! F1={best_dev_f1:.4f}")

            self.history.append({
                "epoch": epoch, "train_loss": train_loss,
                "dev_f1": dev_f1, "elapsed": round(elapsed, 1),
            })
            # Free GPU cache sau mỗi epoch
            torch.cuda.empty_cache()

            # Early stopping
            if self.early_stop and self.early_stop.step(dev_f1):
                logger.info(f"  🛑 Early stopping at epoch {epoch}")
                break

        logger.info(f"\nBest: epoch={best_epoch} | dev_f1={best_dev_f1:.4f}")
        return {"best_epoch": best_epoch, "best_dev_f1": best_dev_f1, "history": self.history}
