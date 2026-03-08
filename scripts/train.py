"""
train.py — Entry point training
Usage:
  python scripts/train.py                                   # dùng config mặc định
  python scripts/train.py model.method=roberta_bilstm_crf  # override bất kỳ field
  python scripts/train.py training.epochs=15 training.fp16=true
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
import torch

from src.data_utils import read_conll, build_label_map, make_dataloader, validate_data
from src.models     import build_model, MODEL_BACKBONE
from src.trainer    import Trainer
from src.evaluator  import Evaluator
from src.utils      import set_seed, get_logger, add_file_handler, save_json

logger = get_logger(__name__)


def setup_wandb(cfg):
    """Khởi tạo WandB nếu được bật trong config."""
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
    # ── Load config ───────────────────────────────────────────────
    def main():
    # ── Load config ───────────────────────────────────────────────
    cli_cfg  = OmegaConf.from_cli()
    
    # Cho phép chỉ định config file qua: python train.py config=configs/config_dvsk.yaml
    config_path = cli_cfg.pop("config", "configs/config.yaml")
    
    base_cfg = OmegaConf.load(config_path)
    cfg      = OmegaConf.merge(base_cfg, cli_cfg) 

    method     = cfg.model.method
    output_dir = os.path.join(cfg.checkpoint.output_dir, method)
    os.makedirs(output_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────
    add_file_handler(logger, os.path.join(output_dir, "train.log"))
    logger.info(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")

    # ── Reproducibility ───────────────────────────────────────────
    set_seed(cfg.project.seed)

    # ── Data ──────────────────────────────────────────────────────
    train_data = read_conll(cfg.data.train)
    dev_data   = read_conll(cfg.data.dev)
    test_data  = read_conll(cfg.data.test)

    validate_data(train_data, "train")
    validate_data(dev_data,   "dev")
    validate_data(test_data,  "test")

    label2id, id2label = build_label_map(train_data)
    save_json({"label2id": label2id, "id2label": id2label},
              os.path.join(output_dir, "label_map.json"))

    # ── Tokenizer ─────────────────────────────────────────────────
    from transformers import AutoTokenizer
    backbone  = cfg.model.backbone or MODEL_BACKBONE[method]
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    bs    = cfg.training.batch_size
    ml    = cfg.data.max_len
    train_loader = make_dataloader(train_data, tokenizer, label2id, ml, bs, shuffle=True)
    dev_loader   = make_dataloader(dev_data,   tokenizer, label2id, ml, bs, shuffle=False)
    test_loader  = make_dataloader(test_data,  tokenizer, label2id, ml, bs, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────
    OmegaConf.update(cfg, "_num_labels", len(label2id))
    model  = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── WandB ─────────────────────────────────────────────────────
    wandb_run = setup_wandb(cfg)

    # ── Train ─────────────────────────────────────────────────────
    evaluator = Evaluator(model, id2label, device, output_dir)
    trainer   = Trainer(model, cfg, output_dir, wandb_run)
    train_res = trainer.train(train_loader, dev_loader, evaluator)

    # ── Test ──────────────────────────────────────────────────────
    logger.info("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(trainer.ckpt_manager.best_path(), map_location=device))

    test_res  = evaluator.full_report(test_loader, split="test")
    evaluator.confusion_matrix(test_loader, split="test")
    evaluator.error_analysis(test_loader, test_data, split="test")

    # ── Save final results ────────────────────────────────────────
    final = {
        "method":      method,
        "seed":        cfg.project.seed,
        "backbone":    backbone,
        **train_res,
        "test_f1":     test_res["f1"],
        "test_report": test_res["report"],
    }
    save_json(final, os.path.join(output_dir, "results.json"))
    logger.info(f"\n✅ Done. Test F1 = {test_res['f1']:.4f}")
    logger.info(f"   Output → {output_dir}")

    if wandb_run:
        wandb_run.log({"test_f1": test_res["f1"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
