"""
utils.py — Seed control, logging setup, checkpoint manager
"""

import os
import random
import logging
import json
import numpy as np
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ── LOGGING ───────────────────────────────────────────────────────────────────
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def add_file_handler(logger: logging.Logger, log_path: str):
    """Add file handler after output_dir is known."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


# ── SEED ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CHECKPOINT MANAGER ────────────────────────────────────────────────────────
class CheckpointManager:
    """
    Giữ top-k checkpoint tốt nhất theo metric (mặc định: dev_f1, cao hơn tốt hơn).
    """

    def __init__(self, output_dir: str, method: str, save_top_k: int = 1,
                 mode: str = "max"):
        self.output_dir = Path(output_dir)
        self.method     = method
        self.save_top_k = save_top_k
        self.mode       = mode          # "max" hoặc "min"
        self.checkpoints: list[dict]    = []   # [{score, path}]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_better(self, score: float, other: float) -> bool:
        return score > other if self.mode == "max" else score < other

    def save(self, model: torch.nn.Module, score: float, epoch: int) -> bool:
        """Lưu nếu tốt hơn checkpoint hiện tại. Trả về True nếu đã lưu."""
        ckpt_path = self.output_dir / f"{self.method}_epoch{epoch:02d}_f1{score:.4f}.pt"
        torch.save(model.state_dict(), ckpt_path)
        self.checkpoints.append({"score": score, "path": str(ckpt_path), "epoch": epoch})

        # Sắp xếp: tốt nhất lên đầu
        reverse = (self.mode == "max")
        self.checkpoints.sort(key=lambda x: x["score"], reverse=reverse)

        # Xóa checkpoint kém hơn nếu vượt quá save_top_k
        while len(self.checkpoints) > self.save_top_k:
            worst = self.checkpoints.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])

        # Tạo symlink "best" cho tiện load
        best_path = self.output_dir / f"{self.method}_best.pt"
        if best_path.exists() or best_path.is_symlink():
            best_path.unlink()
        if self.checkpoints and os.path.exists(self.checkpoints[0]["path"]):  # ← thêm check
            shutil.copy(self.checkpoints[0]["path"], best_path)

        return str(ckpt_path) == self.checkpoints[0]["path"]

    def best_score(self) -> float:
        if not self.checkpoints:
            return float("-inf") if self.mode == "max" else float("inf")
        return self.checkpoints[0]["score"]

    def best_path(self) -> Optional[str]:
        return str(self.output_dir / f"{self.method}_best.pt")


# ── EARLY STOPPING ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best       = float("-inf") if mode == "max" else float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Trả về True nếu nên dừng training."""
        improved = (score > self.best + self.min_delta) if self.mode == "max" \
                   else (score < self.best - self.min_delta)
        if improved:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── MISC ──────────────────────────────────────────────────────────────────────
def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # use NumpyEncoder to handle numpy types/arrays
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
