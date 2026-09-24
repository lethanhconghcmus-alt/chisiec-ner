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
from src.evaluator_extended import extended_test_report
from src.utils      import set_seed, get_logger, add_file_handler, save_json
from src.bioes_utils import build_bioes_label_map, convert_dataset_bio_to_bioes
from src.audit_utils import _file_checksum, bio_to_entities

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
    with open(os.path.join(output_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ── Reproducibility ───────────────────────────────────────────
    set_seed(cfg.project.seed)

    # ── Data ──────────────────────────────────────────────────────
    train_data = read_conll(cfg.data.train)
    dev_data   = read_conll(cfg.data.dev)
    test_data  = read_conll(cfg.data.test)

    validate_data(train_data, "train")
    validate_data(dev_data,   "dev")
    validate_data(test_data,  "test")

    # ── Dataset manifest checksum + sentence/entity counts (mục B) ──────────
    dataset_manifest = {
        "checksums_md5": {
            "train": _file_checksum(cfg.data.train),
            "dev": _file_checksum(cfg.data.dev),
            "test": _file_checksum(cfg.data.test),
        },
        "paths": {"train": cfg.data.train, "dev": cfg.data.dev, "test": cfg.data.test},
    }
    for name, data in (("train", train_data), ("dev", dev_data), ("test", test_data)):
        n_entities = sum(len(bio_to_entities(toks, labs)) for toks, labs in data)
        dataset_manifest[f"n_sentences_{name}"] = len(data)
        dataset_manifest[f"n_entities_{name}"] = n_entities
        logger.info(f"[{name}] {len(data)} sentences, {n_entities} entities")
    save_json(dataset_manifest, os.path.join(output_dir, "dataset_manifest.json"))

    # ── label_scheme (mục A) ─────────────────────────────────────────────
    # Mặc định "bio" = hành vi gốc, KHÔNG đổi gì (M0 baseline). "bioes":
    # convert data đọc được (đang ở BIO) sang BIOES trước khi build dataset
    # -- dùng cho M1 (BIOES, vẫn model CRF cũ, không boundary heads) và bước
    # tiền xử lý chung cho M2 (script riêng scripts/train_boundary.py).
    label_scheme = str(getattr(cfg.data, "label_scheme", "bio") or "bio").lower()
    bioes_mode   = str(getattr(cfg.data, "bioes_mode", "strict") or "strict").lower()
    eval_scheme  = None

    if label_scheme == "bioes":
        train_data, train_conv_stats = convert_dataset_bio_to_bioes(train_data, mode=bioes_mode)
        dev_data,   dev_conv_stats   = convert_dataset_bio_to_bioes(dev_data,   mode=bioes_mode)
        test_data,  test_conv_stats  = convert_dataset_bio_to_bioes(test_data,  mode=bioes_mode)
        save_json(
            {"train": train_conv_stats, "dev": dev_conv_stats, "test": test_conv_stats},
            os.path.join(output_dir, "bioes_conversion_report.json"),
        )
        label2id, id2label = build_bioes_label_map()
        eval_scheme = "IOBES"
    elif label_scheme == "bio":
        label2id, id2label = build_label_map(train_data)
    else:
        raise ValueError(f"Unknown data.label_scheme: {label_scheme!r} (expect 'bio' or 'bioes')")

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

    # ── Constrained BIO decode (opt-in, mac dinh TAT -- xem
    # artifacts/benchmark_v2/bio_decoding_policy.md). CHI benchmark_v2
    # configs (R0/R1/R2) bat evaluation.constrained_decode=true; moi
    # experiment khac (M0/M1/M2/DAPT/...) KHONG truyen field nay -> hanh vi
    # CU (raw unconstrained decode) giu nguyen 100%, khong doi reproducibility.
    constrained_decode = bool(OmegaConf.select(cfg, "evaluation.constrained_decode", default=False))
    constrained_scheme = str(OmegaConf.select(cfg, "evaluation.constrained_scheme", default="bio"))
    decode_policy = "constrained" if constrained_decode else "raw_unconstrained"
    logger.info(f"Decode policy: {decode_policy}"
                + (f" (scheme={constrained_scheme})" if constrained_decode else ""))

    # ── Log seed/batch/steps (yeu cau: seed, dataset manifest checksum,
    # batch/effective batch size, steps/epoch, total planned steps, decode
    # policy -- KHONG co gradient accumulation/multi-device trong pipeline
    # nay nen effective batch size == batch size, xem src/trainer.py) ─────
    import math
    steps_per_epoch = math.ceil(len(train_data) / bs)
    total_planned_steps = steps_per_epoch * cfg.training.epochs
    run_manifest = {
        "seed": cfg.project.seed,
        "dataset_checksums_md5": dataset_manifest["checksums_md5"],
        "batch_size": bs, "effective_batch_size": bs, "gradient_accumulation_steps": 1,
        "steps_per_epoch": steps_per_epoch, "max_epochs": cfg.training.epochs,
        "total_planned_steps": total_planned_steps,
        "decode_policy": decode_policy, "constrained_scheme": constrained_scheme if constrained_decode else None,
    }
    logger.info(f"Run manifest: {run_manifest}")
    save_json(run_manifest, os.path.join(output_dir, "run_manifest.json"))

    # ── Train ─────────────────────────────────────────────────────
    evaluator = Evaluator(model, id2label, device, output_dir, scheme=eval_scheme,
                           use_constrained_decode=constrained_decode, label2id=label2id,
                           constrained_scheme=constrained_scheme)
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

    # ── Extended report (mục C: seen/unseen, partial-span/boundary,
    # ORG<->TITLE confusion, invalid BIO count, GR-11 DTM examples) --
    # CHỈ chạy khi label_scheme=bio (evaluator_extended dùng bio_to_entities,
    # KHÔNG hỗ trợ BIOES) và max_len đủ lớn để không cắt câu (assert bên
    # trong extended_test_report tự chặn nếu lệch số câu).
    extended_report = None
    if label_scheme == "bio":
        extended_report = extended_test_report(
            model, id2label, device, output_dir, test_loader, test_data, train_data, split="test",
            constrained_crf=evaluator.constrained_crf if constrained_decode else None,
        )
    else:
        logger.warning("label_scheme != 'bio' -- BỎ QUA extended_test_report (chỉ hỗ trợ BIO).")

    if extended_report is not None:
        bv = extended_report["bio_violations"]
        logger.info(
            f"BIO violations -- raw={bv['raw_total_violations']} "
            f"(sentences={bv['raw_sentences_with_violations']}) | "
            f"constrained={bv['constrained_total_violations']} "
            f"(sentences={bv['constrained_sentences_with_violations']}) | "
            f"decode_policy={extended_report['decode_policy']}"
        )

    # ── Save final results ────────────────────────────────────────
    from src.utils import count_parameters
    train_time = sum(h.get("elapsed", 0.0) for h in train_res.get("history", []))
    final = {
        "method":       method,
        "seed":         cfg.project.seed,
        "backbone":     backbone,
        "label_scheme": label_scheme,
        **train_res,
        "test_f1":       test_res["f1"],
        "test_report":   test_res["report"],
        "dataset_manifest": dataset_manifest,
        "run_manifest": run_manifest,
        "extended_report_summary": None if extended_report is None else {
            "decode_policy": extended_report["decode_policy"],
            "span_category_counts": extended_report["span_category_counts"],
            "org_title_confusion": extended_report["org_title_confusion"],
            "seen_unseen_f1": extended_report["seen_unseen_f1"],
            "bio_violations": extended_report["bio_violations"],
        },
        "total_params":  count_parameters(model),
        "train_time":    round(train_time, 1),
    }
    save_json(final, os.path.join(output_dir, "results.json"))
    logger.info(f"\n✅ Done. Test F1 = {test_res['f1']:.4f}")
    logger.info(f"   Output → {output_dir}")

    if wandb_run:
        wandb_run.log({"test_f1": test_res["f1"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
