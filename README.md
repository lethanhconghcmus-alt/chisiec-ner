# Ancient Chinese NER

Sentence segmentation (断句) and Named Entity Recognition (NER) for Ancient Chinese text using GuwenBERT + CRF.

| | |
|---|---|
| **Backbone** | `ethanyt/guwenbert-base` |
| **Architecture** | BERT + CRF |
| **Entity Types** | `PER` `LOC` `TITLE` `DTM` `ORG` |
| **Training Data** | AnChineseNERE · CHisIEC · BEEVETA · C-CLUE |
| **Segmentation Model** | `raynardj/classical-chinese-punctuation-guwen-biaodian` |

---

## Project Structure

```
├── api/
│   ├── main.py           # FastAPI app
│   ├── predictor.py      # NER inference logic
│   └── segmenter.py      # Sentence segmentation (punctuation model)
├── src/
│   ├── models.py         # GuwenBertCRF model
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

> **Model weights (`best.pt`) are NOT stored in this repository.**  
> They are loaded from an external storage location (e.g. server volume).
> The segmentation model is downloaded from HuggingFace on first startup.

Expected runtime paths:

```
/app/checkpoints/best.pt
/app/artifacts/label_map.json
```

---

## Deploy (server)

### 1. Get the code

```bash
git clone https://github.com/clc-hcmus-edu-vn/ancient-chinese-ner.git
cd ancient-chinese-ner
```

### 2. Get the model weights

`best.pt` is attached to the GitHub Release [`model-v1.0`](https://github.com/clc-hcmus-edu-vn/ancient-chinese-ner/releases/tag/model-v1.0)
(repo members only):

```bash
mkdir -p checkpoints
gh release download model-v1.0 -R clc-hcmus-edu-vn/ancient-chinese-ner -p best.pt -D checkpoints/
```

Or download `best.pt` from the release page in a browser and put it at `checkpoints/best.pt`.

### 3. Start

```bash
docker compose up -d --build
docker compose logs -f          # first start downloads ~1.2GB of HuggingFace models
curl localhost:9001/health      # wait for model_loaded and segmenter_loaded = true
```

The API is served on port **9001** (`http://<server>:9001`, Swagger UI at `/docs`).
HuggingFace models are cached in the `hf-cache` volume, so restarts do not re-download them.
The server needs internet access on first start, and about 3GB RAM.

### Update

```bash
git pull
docker compose up -d --build
```

---

## Run API locally

```bash
CHECKPOINT_PATH=checkpoints/best.pt LABEL_MAP_PATH=artifacts/label_map.json \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "segmenter_loaded": true }
```

### `POST /predict`

**Request**

```json
{
  "sentences": "命胡士楊、阮名實、阮廷正等徃關上候命。"
}
```

**Response**

```json
{
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

Sentence segmentation + NER in one call. Consecutive sentences are joined into
windows of up to `MAX_LEN` characters before NER, so short sentences keep their
context and long ones are not truncated. Entity offsets are relative to each sentence.

Set `"auto_segment": false` to keep the input punctuation (split on newlines and `。？！；`).

**Request**

```json
{
  "text": "命胡士楊阮名實阮廷正等徃關上候命以其徃關上接領北朝頒賞勑諭銀幣故也",
  "auto_segment": true
}
```

**Response**

```json
{
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
| `BACKBONE` | `ethanyt/guwenbert-base` | HuggingFace model |
| `CHECKPOINT_PATH` | `/app/checkpoints/best.pt` | Model weights path |
| `LABEL_MAP_PATH` | `/app/artifacts/label_map.json` | Label mapping |
| `MAX_LEN` | `128` | Max sequence length |
| `PUNC_MODEL` | `raynardj/classical-chinese-punctuation-guwen-biaodian` | Segmentation model |
| `PUNC_WINDOW` | `400` | Characters per segmentation pass |
| `MAX_INPUT` | `20000` | Max characters for `/segment` and `/analyze` |
| `DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |

---

## Notes

- `best.pt` must match `label_map.json`
- Model weights are loaded from a mounted volume - not included in this repository (see GitHub Releases)
- Repository contains only code and lightweight artifacts
