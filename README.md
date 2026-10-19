# Ancient Chinese NER

Sentence segmentation (断句) and Named Entity Recognition (NER) for Ancient Chinese text,
with a choice of BERT encoder + CRF models.

| | |
|---|---|
| **Architecture** | BERT encoder + Linear + CRF (BIO) |
| **Entity Types** | `PER` `LOC` `TITLE` `DTM` `ORG` |
| **Segmentation Model** | `raynardj/classical-chinese-punctuation-guwen-biaodian` |

### NER models

| `model` | Backbone | Training data | Test F1 | Checkpoint file |
|---|---|---|---|---|
| `guwenbert-crf-tq` (default) | `ethanyt/guwenbert-base` | AnChineseNERE · CHisIEC · BEEVETA · C-CLUE (Chinese histories) | 0.932 (TQ test); 0.41 on DVSKTT (not trained on it) | `best.pt` |
| `sikubert-crf-dvsktt` | `SIKU-BERT/sikubert` | DVSKTT `dataset_v2_clean` (1,142 / 147 / 157 sentences) | **0.678** (DVSKTT test) | `sikubert_crf_dvsktt.pt` |
| `gujiroberta-crf-dvsktt` | `hsc748NLP/GujiRoBERTa_jian` | DVSKTT `dataset_v1` (1,277 / 160 / 160 sentences) | 0.663 (DVSKTT test) | `gujiroberta_crf_dvsktt.pt` |

For Vietnamese historical text (Đại Việt sử ký toàn thư), use `sikubert-crf-dvsktt`.
F1 = entity-level exact match. The two DVSKTT models are evaluated on different test splits.

---

## Project Structure

```
├── api/
│   ├── main.py           # FastAPI app
│   ├── predictor.py      # NER models (registry) + inference logic
│   └── segmenter.py      # Sentence segmentation (punctuation model)
├── src/
│   ├── models.py         # BERT + CRF model
│   └── data_utils.py     # Dataset + preprocessing
├── scripts/
│   ├── train.py          # Training script
│   └── inference.py      # Load model + predict
├── configs/
│   └── config.yaml
├── artifacts/
│   └── label_map.json    # Label mapping (kept in repo)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Model & Deployment

> **Model weights are NOT stored in this repository.**  
> They are loaded from an external storage location (e.g. server volume).
> Backbones and the segmentation model are downloaded from HuggingFace on first use.

Expected runtime paths (only the checkpoints you put there become available):

```
/app/checkpoints/best.pt                     # guwenbert-crf-tq
/app/checkpoints/sikubert_crf_dvsktt.pt      # sikubert-crf-dvsktt
/app/checkpoints/gujiroberta_crf_dvsktt.pt   # gujiroberta-crf-dvsktt
/app/artifacts/label_map.json                # shared by all models
```

---

## Deploy (server)

### 1. Get the code

```bash
git clone https://github.com/clc-hcmus-edu-vn/ancient-chinese-ner.git
cd ancient-chinese-ner
```

### 2. Get the model weights

Checkpoints are attached to GitHub Releases (repo members only):
[`model-v1.0`](https://github.com/clc-hcmus-edu-vn/ancient-chinese-ner/releases/tag/model-v1.0) (GuwenBERT) and
[`model-v2.0`](https://github.com/clc-hcmus-edu-vn/ancient-chinese-ner/releases/tag/model-v2.0) (SikuBERT, GujiRoBERTa).

```bash
mkdir -p checkpoints
R=clc-hcmus-edu-vn/ancient-chinese-ner
gh release download model-v1.0 -R $R -p best.pt -D checkpoints/
gh release download model-v2.0 -R $R -p sikubert_crf_dvsktt.pt -p gujiroberta_crf_dvsktt.pt -D checkpoints/
```

Or download them from the release pages in a browser into `checkpoints/`.
Set the default model with `NER_MODEL` in `docker-compose.yml`.

### 3. Start

```bash
docker compose up -d --build
docker compose logs -f          # first start downloads the backbone + segmentation models
curl localhost:9001/health      # wait for model_loaded and segmenter_loaded = true
```

The API is served on port **9001** (`http://<server>:9001`, Swagger UI at `/docs`).
HuggingFace models are cached in the `hf-cache` volume, so restarts do not re-download them.
The server needs internet access on first start, about 3GB RAM, plus ~0.8GB per extra NER model loaded.

### Update

```bash
git pull
docker compose up -d --build
```

---

## Web so sánh (chạy local)

Giao diện để thử tách câu + NER và so sánh nhiều model trên cùng một đoạn Hán cổ:
chọn model ngắt câu (`guwen-punc` / `biaodian` / giữ nguyên), chọn nhiều model NER,
xem âm Hán Việt dưới mỗi câu, đối chiếu với true label, và chấm P/R/F1 trên 54 đoạn gold.

```bash
git clone -b feature/segment-api https://github.com/lethanhconghcmus-alt/chisiec-ner.git
cd chisiec-ner
pip install -r requirements.txt

# model NER (GuwenBERT+CRF · TQ, 397MB)
mkdir -p checkpoints
curl -L -o checkpoints/best.pt \
  https://github.com/lethanhconghcmus-alt/chisiec-ner/releases/download/model-v1.0/best.pt

./run_web.sh                 # → http://localhost:8000   (đổi cổng: PORT=8010 ./run_web.sh)
```

Windows: chạy `python -m uvicorn web.app:app --port 8000` thay cho `./run_web.sh`.

- Lần đầu chạy sẽ tải các model từ HuggingFace (GuwenBERT, guwen-punc, guwen-ner… ~1.6GB),
  mỗi model chỉ tải khi được chọn lần đầu. Cần ~4GB RAM; Mac M-series dùng MPS, có NVIDIA thì dùng CUDA.
- **Đổi phồn → giản thể trước NER**: GuwenBERT có vocab giản thể nên ~24% chữ DVSKTT phồn thể
  thành `[UNK]`; bật tuỳ chọn này tăng F1 gold từ 45.6 lên 52.5 mà không cần train lại.
- Web dùng chung danh sách model với API (`api/predictor.py`): có checkpoint nào trong `checkpoints/`
  thì model đó hiện lên (xem bảng NER models ở đầu README), cộng thêm `guwen-ner` công khai.
- Cột "F1 chưa thấy" cần file train DVSKTT (`DVSK_TRAIN=...`), không có thì hiện "—".
- True label: `web/gold_manual.txt` (mỗi dòng 1 đoạn, thực thể dạng `{chữ|NHÃN}`), sửa xong F5 là cập nhật.
- Link chia sẻ: `?manual=3&simp=1&hv=ruby&models=guwenbert-crf-tq,sikubert-crf-dvsktt` (mở đoạn gold số 3, bật giản thể,
  âm HV dưới từng chữ), `?tab=eval&run=1` (chấm ngay).

---

## Run API locally

```bash
NER_MODEL=sikubert-crf-dvsktt uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "segmenter_loaded": true,
  "default_model": "guwenbert-crf-tq", "loaded_models": ["guwenbert-crf-tq"] }
```

### `GET /models`

NER models whose checkpoint is present (the default one is loaded at startup, others on first request).

```json
{
  "default": "guwenbert-crf-tq",
  "models": [
    { "name": "guwenbert-crf-tq", "backbone": "ethanyt/guwenbert-base", "max_len": 128, "description": "…", "loaded": true },
    { "name": "sikubert-crf-dvsktt", "backbone": "SIKU-BERT/sikubert", "max_len": 256, "description": "…", "loaded": false }
  ]
}
```

### `POST /predict`

**Request**

One sentence per line. `model` is optional (default: `NER_MODEL`).

```json
{
  "sentences": "命胡士楊、阮名實、阮廷正等徃關上候命。",
  "model": "sikubert-crf-dvsktt"
}
```

**Response**

```json
{
  "model": "sikubert-crf-dvsktt",
  "data": [
    {
      "text": "命胡士楊、阮名實、阮廷正等徃關上候命。",
      "entities": [
        { "text": "胡士楊", "label": "PER", "start": 1, "end": 4 },
        { "text": "阮名實", "label": "PER", "start": 5, "end": 8 },
        { "text": "阮廷正", "label": "PER", "start": 9, "end": 12 }
      ]
    }
  ]
}
```

### `POST /segment`

Sentence segmentation. Existing punctuation is removed and re-predicted by the model.

**Request**

```json
{
  "text": "命胡士楊阮名實阮廷正等徃關上候命以其徃關上接領北朝頒賞勑諭銀幣故也"
}
```

**Response**

```json
{
  "punctuated": "命胡士楊。阮名實。阮廷正等徃關上候命。以其徃關上接領北朝頒賞勑諭銀幣故也。",
  "sentences": [
    "命胡士楊。",
    "阮名實。",
    "阮廷正等徃關上候命。",
    "以其徃關上接領北朝頒賞勑諭銀幣故也。"
  ]
}
```

### `POST /analyze`

Sentence segmentation + NER in one call (optional `model`, as in `/predict`).
Consecutive sentences are joined into windows of up to the model's max length before NER, so short sentences keep their
context and long ones are not truncated. Entity offsets are relative to each sentence.

Set `"auto_segment": false` to keep the input punctuation (split on newlines and `。？！；`).

**Request**

```json
{
  "text": "命胡士楊阮名實阮廷正等徃關上候命以其徃關上接領北朝頒賞勑諭銀幣故也",
  "auto_segment": true,
  "model": "guwenbert-crf-tq"
}
```

**Response**

```json
{
  "model": "guwenbert-crf-tq",
  "punctuated": "命胡士楊。阮名實。阮廷正等徃關上候命。以其徃關上接領北朝頒賞勑諭銀幣故也。",
  "data": [
    { "text": "命胡士楊。", "entities": [{ "text": "胡士楊", "label": "PER", "start": 1, "end": 4 }] },
    { "text": "阮名實。", "entities": [{ "text": "阮名實", "label": "PER", "start": 0, "end": 3 }] },
    { "text": "阮廷正等徃關上候命。", "entities": [{ "text": "阮廷正", "label": "PER", "start": 0, "end": 3 }] },
    { "text": "以其徃關上接領北朝頒賞勑諭銀幣故也。", "entities": [{ "text": "北朝", "label": "LOC", "start": 7, "end": 9 }] }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NER_MODEL` | `guwenbert-crf-tq` | Default NER model (see table above) |
| `CHECKPOINT_DIR` | `checkpoints` | Folder holding the model checkpoints |
| `CHECKPOINT_PATH` | `<CHECKPOINT_DIR>/best.pt` | Checkpoint of `guwenbert-crf-tq` |
| `LABEL_MAP_PATH` | `artifacts/label_map.json` | Label mapping (shared by all models) |
| `PUNC_MODEL` | `raynardj/classical-chinese-punctuation-guwen-biaodian` | Segmentation model |
| `PUNC_WINDOW` | `400` | Characters per segmentation pass |
| `MAX_INPUT` | `20000` | Max characters for `/segment` and `/analyze` |
| `DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |

---

## Notes

- All checkpoints use the same 11-label BIO map in `artifacts/label_map.json`
- Characters the tokenizer drops (e.g. Private Use Area code points, common in DVSKTT) are labelled `O`;
  labels of the following characters stay aligned
- Model weights are loaded from a mounted volume - not included in this repository (see GitHub Releases)
- Repository contains only code and lightweight artifacts
