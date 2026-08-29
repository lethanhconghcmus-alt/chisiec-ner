"""
train.py — Entry point training Ancient Chinese NER
Usage:
  python scripts/train.py                                      # dùng config mặc định
  python scripts/train.py model.method=guwenbert_linear       # override method
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
    cli_cfg     = OmegaConf.from_cli()
    config_path = cli_cfg.pop("config", "configs/config.yaml")
    base_cfg    = OmegaConf.load(config_path)
    cfg         = OmegaConf.merge(base_cfg, cli_cfg)

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
    save_json(
        {"label2id": label2id, "id2label": {str(i): l for i, l in id2label.items()}},
        os.path.join(output_dir, "label_map.json"),
    )

    # ── Tokenizer ─────────────────────────────────────────────────
    from transformers import AutoTokenizer
    backbone  = cfg.model.backbone or MODEL_BACKBONE[method]
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    # ── Gazetteer (feature phụ, tùy chọn) ────────────────────────────
    # multihot=True (mặc định): mỗi vị trí có nhiều cờ độc lập (TITLE/ORG/LOC/
    # PER/DTM có thể cùng =1), không ép chọn 1 loại duy nhất khi mơ hồ -- xem
    # src/gazetteer.py:tag_multihot. multihot=False: giữ hành vi cũ (1 nhãn
    # categorical/vị trí, ưu tiên cứng theo thứ tự loại).
    gaz_tagger = None
    gaz_multihot = False
    gaz_types = None
    if getattr(cfg, "gazetteer", None) and cfg.gazetteer.enabled:
        from src.gazetteer import load_gazetteer, GazetteerTagger, GAZ_LABELS, GAZ_TYPES
        surfaces     = load_gazetteer(cfg.gazetteer.dir)
        gaz_tagger   = GazetteerTagger(surfaces)
        gaz_multihot = bool(getattr(cfg.gazetteer, "multihot", True))
        cfg_types    = getattr(cfg.gazetteer, "types", None)
        gaz_types    = list(cfg_types) if cfg_types else list(GAZ_TYPES)
        if gaz_multihot:
            OmegaConf.update(cfg, "model.gaz_input_dim", len(gaz_types))
            logger.info(f"Gazetteer feature enabled (multi-hot): {gaz_types}")
        else:
            OmegaConf.update(cfg, "model.gaz_vocab_size", len(GAZ_LABELS))
            logger.info(f"Gazetteer feature enabled (categorical, legacy): {len(GAZ_LABELS)} tags")

    bs = cfg.training.batch_size
    ml = cfg.data.max_len
    train_loader = make_dataloader(train_data, tokenizer, label2id, ml, bs, shuffle=True,  gaz_tagger=gaz_tagger, gaz_multihot=gaz_multihot, gaz_types=gaz_types)
    dev_loader   = make_dataloader(dev_data,   tokenizer, label2id, ml, bs, shuffle=False, gaz_tagger=gaz_tagger, gaz_multihot=gaz_multihot, gaz_types=gaz_types)
    test_loader  = make_dataloader(test_data,  tokenizer, label2id, ml, bs, shuffle=False, gaz_tagger=gaz_tagger, gaz_multihot=gaz_multihot, gaz_types=gaz_types)

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
    model.load_state_dict(
        torch.load(trainer.ckpt_manager.best_path(), map_location=device)
    )

    test_res  = evaluator.full_report(test_loader,  split="test")
    evaluator.confusion_matrix(test_loader,  split="test")
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
