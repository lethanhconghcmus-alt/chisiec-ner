"""
inference.py — Load model đã train và predict câu mới
Usage:
  # Predict 1 câu
  python scripts/inference.py --text "唐太宗李世民貞觀元年"

  # Predict file CoNLL
  python scripts/inference.py --input_file data/ancient/test.txt --output_file outputs/predictions.txt

  # Chỉ định method khác
  python scripts/inference.py --method guwenbert_linear --text "命胡士楊徃關上候命"
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.models    import build_model, MODEL_BACKBONE
from src.data_utils import read_conll
from src.utils     import load_json, get_logger

logger = get_logger(__name__)


def load_trained_model(method: str, output_dir: str, device: torch.device):
    """Load model + label map từ output directory."""
    model_dir = os.path.join(output_dir, method)
    label_map = load_json(os.path.join(model_dir, "label_map.json"))
    label2id  = label_map["label2id"]
    id2label  = {int(k): v for k, v in label_map["id2label"].items()}

    cfg = OmegaConf.create({
        "model": {
            "method":       method,
            "backbone":     None,
            "lstm_hidden":  256,
            "lstm_layers":  2,
            "kan_hidden":   512,
            "kan_knots":    8,
            "dropout":      0.1,
            "pretrained_ckpt": None,
        },
        "_num_labels": len(label2id),
    })

    model = build_model(cfg)
    ckpt  = os.path.join(model_dir, f"{method}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    logger.info(f"Loaded model from {ckpt}")
    return model, label2id, id2label


@torch.no_grad()
def predict_sentence(
    tokens:    list,
    model,
    tokenizer,
    label2id:  dict,
    id2label:  dict,
    max_len:   int = 128,
    device:    torch.device = torch.device("cpu"),
) -> list:
    """Predict NER tags cho 1 câu. Returns list of (token, label)."""
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = torch.zeros_like(input_ids).to(device)

    preds = model(input_ids, attention_mask, token_type_ids)
    # Handle both CRF (list of lists) and Linear (tensor) output
    if isinstance(preds, list):
        pred_ids = preds[0]
    else:
        pred_ids = preds[0].tolist()

    word_ids    = enc.word_ids()
    token_preds = {}
    prev_word   = None
    for pos, (wid, pred) in enumerate(zip(word_ids, pred_ids)):
        if wid is None or wid == prev_word:
            prev_word = wid
            continue
        token_preds[wid] = id2label.get(pred, "O")
        prev_word = wid

    return [(tok, token_preds.get(i, "O")) for i, tok in enumerate(tokens)]


def entities_from_bio(token_label_pairs: list) -> list:
    """Extract entity spans từ BIO tags."""
    entities, current = [], None
    for token, label in token_label_pairs:
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": label[2:], "tokens": [token]}
        elif label.startswith("I-") and current and current["type"] == label[2:]:
            current["tokens"].append(token)
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return entities


def predict_file(input_path, output_path, model, tokenizer, label2id, id2label, device):
    data = read_conll(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        for tokens, _ in data:
            preds = predict_sentence(
                tokens, model, tokenizer, label2id, id2label, device=device
            )
            for token, pred_label in preds:
                f.write(f"{token}\t{pred_label}\n")
            f.write("\n")
    logger.info(f"Predictions saved → {output_path}")


def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone  = MODEL_BACKBONE[args.method]
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model, label2id, id2label = load_trained_model(args.method, args.output_dir, device)

    if args.text:
        tokens   = list(args.text)   # character-level
        result   = predict_sentence(tokens, model, tokenizer, label2id, id2label, device=device)
        entities = entities_from_bio(result)

        print(f"\nInput : {args.text}")
        print(f"\nToken-level predictions:")
        print(f"  {'Token':<8} {'Label'}")
        print(f"  {'─'*20}")
        for tok, lbl in result:
            marker = "◀" if lbl != "O" else ""
            print(f"  {tok:<8} {lbl} {marker}")

        if entities:
            print(f"\nEntities found:")
            for ent in entities:
                print(f"  [{ent['type']}] {''.join(ent['tokens'])}")
        else:
            print("\nNo entities found.")

    elif args.input_file:
        predict_file(
            args.input_file, args.output_file,
            model, tokenizer, label2id, id2label, device,
        )
    else:
        print("Provide --text or --input_file")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method",     default="guwenbert_crf",
                   choices=list(MODEL_BACKBONE.keys()))
    p.add_argument("--output_dir", default="outputs/ancient")
    p.add_argument("--text",       default=None, help="Câu cần predict")
    p.add_argument("--input_file", default=None, help="File CoNLL cần predict")
    p.add_argument("--output_file", default="outputs/ancient/predictions.txt")
    main(p.parse_args())
