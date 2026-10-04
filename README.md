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

## Run API

### Local

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t ancient-ner .

docker run -p 8000:8000 \
  -v /path/to/checkpoints:/app/checkpoints \
  ancient-ner
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
- Model weights are loaded from a mounted volume - not included in this repository
- Repository contains only code and lightweight artifacts
