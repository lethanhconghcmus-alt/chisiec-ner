from scripts.inference import load_model_and_assets, predict_text


class AncientNERPredictor:
    def __init__(
        self,
        method: str = "guwenbert_crf",
        checkpoint_path: str = "checkpoints/best.pt",
        label_map_path: str = "artifacts/label_map.json",
    ):
        self.method = method
        self.model, self.tokenizer, _, self.id2label, self.device = load_model_and_assets(
            method=method,
            checkpoint_path=checkpoint_path,
            label_map_path=label_map_path,
        )

    def predict(self, text: str):
        return predict_text(
            text=text,
            model=self.model,
            tokenizer=self.tokenizer,
            id2label=self.id2label,
            device=self.device,
        )
