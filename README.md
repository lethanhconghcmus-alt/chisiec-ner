# Ancient Chinese NER

A Named Entity Recognition (NER) system for Ancient Chinese text using GuwenBERT + CRF.

| | |
|---|---|
| **Backbone** | `ethanyt/guwenbert-base` |
| **Architecture** | BERT + CRF |
| **Entity Types** | `PER` `LOC` `TITLE` `DTM` `ORG` |
| **Training Data** | AnChineseNERE · CHisIEC · BEEVETA · C-CLUE |

---

## Project Structure

```
├── api/
│   ├── main.py           # FastAPI app
│   └── predictor.py      # Inference logic
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
{ "status": "ok", "model_loaded": true }
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

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKBONE` | `ethanyt/guwenbert-base` | HuggingFace model |
| `CHECKPOINT_PATH` | `/app/checkpoints/best.pt` | Model weights path |
| `LABEL_MAP_PATH` | `/app/artifacts/label_map.json` | Label mapping |
| `MAX_LEN` | `128` | Max sequence length |

---

## Notes

- `best.pt` must match `label_map.json`
- Model weights are loaded from a mounted volume - not included in this repository
- Repository contains only code and lightweight artifacts
