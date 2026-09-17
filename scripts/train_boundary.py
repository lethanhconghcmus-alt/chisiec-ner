"""
train_boundary.py — entry point train M2: backbone + CRF (BIOES) + start/end
boundary heads (BertCRFBoundaryNER). Không đụng scripts/train.py (M0/M1 vẫn
chạy nguyên như cũ qua script đó).

Usage:
  python scripts/train_boundary.py                                            # config mặc định
  python scripts/train_boundary.py config=configs/config_boundary.yaml \
      model.backbone=SIKU-BERT/sikubert model.enable_boundary_auxiliary=true
  python scripts/train_boundary.py model.enable_boundary_auxiliary=false      # tắt boundary loss (vẫn BIOES + CRF)
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
import torch

from src.data_utils import read_conll, make_dataloader, validate_data
from src.models import build_model
from src.trainer_boundary import BoundaryTrainer
from src.evaluator_boundary import BoundaryEvaluator
from src.utils import set_seed, get_logger, add_file_handler, save_json
from src.bioes_utils import (
    build_bioes_label_map,
    convert_dataset_bio_to_bioes,
    compute_boundary_pos_weight,
)

logger = get_logger(__name__)


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def setup_wandb(cfg):
    if not cfg.logging.use_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=f"{cfg.model.method}_{cfg.project.name}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        logger.info(f"WandB run: {run.url}")
        return run
    except ImportError:
        logger.warning("wandb not installed, skipping")
        return None


def main():
    cli_cfg     = OmegaConf.from_cli()
    config_path = cli_cfg.pop("config", "configs/config_boundary.yaml")
    base_cfg    = OmegaConf.load(config_path)
    cfg         = OmegaConf.merge(base_cfg, cli_cfg)

    method     = cfg.model.method  # luôn "bert_crf_boundary" trong config này
    output_dir = os.path.join(cfg.checkpoint.output_dir, method)
    os.makedirs(output_dir, exist_ok=True)

    add_file_handler(logger, os.path.join(output_dir, "train.log"))
    logger.info(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Git commit: {_git_commit_hash()}")

    set_seed(cfg.project.seed)

    # ── Data: đọc BIO, validate, convert -> BIOES (mục A) ────────────────
    train_data = read_conll(cfg.data.train)
    dev_data   = read_conll(cfg.data.dev)
    test_data  = read_conll(cfg.data.test)

    validate_data(train_data, "train")
    validate_data(dev_data,   "dev")
    validate_data(test_data,  "test")

    bioes_mode = str(getattr(cfg.data, "bioes_mode", "strict") or "strict").lower()
    train_data, train_conv = convert_dataset_bio_to_bioes(train_data, mode=bioes_mode)
    dev_data,   dev_conv   = convert_dataset_bio_to_bioes(dev_data,   mode=bioes_mode)
    test_data,  test_conv  = convert_dataset_bio_to_bioes(test_data,  mode=bioes_mode)
    save_json(
        {"train": train_conv, "dev": dev_conv, "test": test_conv},
        os.path.join(output_dir, "bioes_conversion_report.json"),
    )

    label2id, id2label = build_bioes_label_map()
    save_json(
        {"label2id": label2id, "id2label": {str(i): l for i, l in id2label.items()}},
        os.path.join(output_dir, "label_map.json"),
    )
    o_label_id = label2id["O"]

    # ── Boundary pos_weight — CHỈ tính trên TRAIN (mục D) ────────────────
    pos_weight_max = float(getattr(cfg.model, "boundary_pos_weight_max", 10.0) or 10.0)
    pw_stats = compute_boundary_pos_weight(train_data, max_pos_weight=pos_weight_max)
    save_json(pw_stats, os.path.join(output_dir, "boundary_pos_weight_stats.json"))

    # ── Tokenizer + dataloader ────────────────────────────────────────────
    from transformers import AutoTokenizer
    backbone = cfg.model.backbone
    if not backbone:
        raise ValueError("model.backbone bắt buộc (ví dụ SIKU-BERT/sikubert, hsc748NLP/GujiRoBERTa_jian)")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    tokenizer.save_pretrained(output_dir)

    bs = cfg.training.batch_size
    ml = cfg.data.max_len
    train_loader = make_dataloader(train_data, tokenizer, label2id, ml, bs, shuffle=True, derive_boundary=True)
    dev_loader   = make_dataloader(dev_data,   tokenizer, label2id, ml, bs, shuffle=False, derive_boundary=True)
    test_loader  = make_dataloader(test_data,  tokenizer, label2id, ml, bs, shuffle=False, derive_boundary=True)

    # ── Model ─────────────────────────────────────────────────────
    OmegaConf.update(cfg, "_num_labels", len(label2id))
    OmegaConf.update(cfg, "_o_label_id", o_label_id)
    if str(getattr(cfg.model, "boundary_loss_type", "weighted_bce")) == "weighted_bce":
        OmegaConf.update(cfg, "_start_pos_weight", pw_stats["start_pos_weight"])
        OmegaConf.update(cfg, "_end_pos_weight", pw_stats["end_pos_weight"])
    with open(os.path.join(output_dir, "full_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    model  = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_run = setup_wandb(cfg)

    # ── Train ─────────────────────────────────────────────────────
    evaluator = BoundaryEvaluator(model, id2label, device, output_dir)
    trainer   = BoundaryTrainer(model, cfg, output_dir, wandb_run)
    train_res = trainer.train(train_loader, dev_loader, evaluator)

    # ── Test (best checkpoint theo strict entity-level dev F1) ──────
    logger.info("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(trainer.ckpt_manager.best_path(), map_location=device))

    test_res = evaluator.full_report(test_loader, split="test")
    evaluator.error_analysis(test_loader, test_data, split="test")

    from src.utils import count_parameters
    train_time = sum(h.get("elapsed", 0.0) for h in train_res.get("history", []))
    final = {
        "method":       method,
        "seed":         cfg.project.seed,
        "backbone":     backbone,
        "label_scheme": "bioes",
        "enable_boundary_auxiliary": bool(cfg.model.enable_boundary_auxiliary),
        "boundary_weight": float(cfg.model.boundary_weight),
        "boundary_loss_type": str(cfg.model.boundary_loss_type),
        "git_commit":   _git_commit_hash(),
        **train_res,
        "test_f1":        test_res["f1"],
        "test_report":    test_res["report"],
        "test_start_f1":  test_res["start_f1"],
        "test_end_f1":    test_res["end_f1"],
        "total_params":   count_parameters(model),
        "train_time":     round(train_time, 1),
    }
    save_json(final, os.path.join(output_dir, "results.json"))
    logger.info(f"\nDone. Test entity-F1 = {test_res['f1']:.4f}")
    logger.info(f"   Output → {output_dir}")

    if wandb_run:
        wandb_run.log({"test_f1": test_res["f1"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
