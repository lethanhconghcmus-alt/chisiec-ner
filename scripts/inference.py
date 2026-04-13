"""
inference.py — Load trained model and predict new text

Usage:
  python3 scripts/inference.py --method guwenbert_crf --text "唐太宗李世民貞觀元年"

  python3 scripts/inference.py \
    --method guwenbert_crf \
    --input_file data/test.txt \
    --output_file outputs/predictions.txt
"""

import argparse
import os
import sys
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.models import build_model, MODEL_BACKBONE
from src.data_utils import read_conll
from src.utils import load_json, get_logger

logger = get_logger(__name__)


def load_model_and_assets(
    method: str = "guwenbert_crf",
    checkpoint_path: str = "checkpoints/best.pt",
    label_map_path: str = "artifacts/label_map.json",
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = load_json(label_map_path)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    cfg = OmegaConf.create({
        "model": {
            "method": method,
            "backbone": None,
            "lstm_hidden": 256,
            "lstm_layers": 2,
            "kan_hidden": 512,
            "kan_knots": 8,
            "dropout": 0.1,
        },
        "_num_labels": len(label2id),
    })

    model = build_model(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    backbone = MODEL_BACKBONE[method]
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Loaded label map from {label_map_path}")

    return model, tokenizer, label2id, id2label, device


@torch.no_grad()
def predict_sentence(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    max_len: int = 128,
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[str, str]]:
    tokens = list(text)

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

    preds = model(input_ids, attention_mask, token_type_ids)[0]

    word_ids = enc.word_ids(batch_index=0)
    token_preds = {}
    prev_word = None

    for wid, pred in zip(word_ids, preds):
        if wid is None or wid == prev_word:
            prev_word = wid
            continue
        token_preds[wid] = id2label.get(int(pred), "O")
        prev_word = wid

    return [(tok, token_preds.get(i, "O")) for i, tok in enumerate(tokens)]


def entities_from_bio(token_label_pairs: List[Tuple[str, str]]) -> List[Dict]:
    entities = []
    current = None

    for idx, (token, label) in enumerate(token_label_pairs):
        if label.startswith("B-"):
            if current is not None:
                current["end"] = idx - 1
                current["text"] = "".join(current["tokens"])
                entities.append(current)

            current = {
                "type": label[2:],
                "tokens": [token],
                "start": idx,
            }

        elif label.startswith("I-") and current is not None and current["type"] == label[2:]:
            current["tokens"].append(token)

        else:
            if current is not None:
                current["end"] = idx - 1
                current["text"] = "".join(current["tokens"])
                entities.append(current)
                current = None

    if current is not None:
        current["end"] = len(token_label_pairs) - 1
        current["text"] = "".join(current["tokens"])
        entities.append(current)

    for ent in entities:
        ent.pop("tokens", None)

    return entities


def predict_text(
    text: str,
    model,
    tokenizer,
    id2label: Dict[int, str],
    device: torch.device,
):
    token_level = predict_sentence(
        text=text,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
    )
    entities = entities_from_bio(token_level)

    return {
        "text": text,
        "tokens": [{"token": tok, "label": lbl} for tok, lbl in token_level],
        "entities": entities,
    }


def predict_file(input_path: str, output_path: str, model, tokenizer, id2label, device):
    data = read_conll(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for tokens, _ in data:
            text = "".join(tokens)
            preds = predict_sentence(text, model, tokenizer, id2label, device=device)
            for token, pred_label in preds:
                f.write(f"{token}\t{pred_label}\n")
            f.write("\n")

    logger.info(f"Predictions saved -> {output_path}")


def main(args):
    model, tokenizer, _, id2label, device = load_model_and_assets(
        method=args.method,
        checkpoint_path=args.checkpoint_path,
        label_map_path=args.label_map_path,
    )

    if args.text:
        result = predict_text(
            text=args.text,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
        )

        print(f"\nInput: {result['text']}")
        print("\nToken-level predictions:")
        print(f"{'Token':<8} Label")
        print("─" * 24)
        for item in result["tokens"]:
            marker = "◀" if item["label"] != "O" else ""
            print(f"{item['token']:<8} {item['label']} {marker}")

        if result["entities"]:
            print("\nEntities found:")
            for ent in result["entities"]:
                print(f"[{ent['type']}] {ent['text']} ({ent['start']}, {ent['end']})")
        else:
            print("\nNo entities found.")

    elif args.input_file:
        predict_file(
            input_path=args.input_file,
            output_path=args.output_file,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
        )
    else:
        print("Provide --text or --input_file")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        default="guwenbert_crf",
        choices=["guwenbert_crf", "roberta_bilstm_crf", "roberta_kan_crf"],
    )
    p.add_argument("--checkpoint_path", default="checkpoints/best.pt")
    p.add_argument("--label_map_path", default="artifacts/label_map.json")
    p.add_argument("--text", default=None, help="Text to predict")
    p.add_argument("--input_file", default=None, help="CoNLL file to predict")
    p.add_argument("--output_file", default="outputs/predictions.txt")
    main(p.parse_args())
