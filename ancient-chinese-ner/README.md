# Ancient Chinese NER

NER pipeline for Ancient Chinese text using GuwenBERT + CRF.

**Trained on:** AnChineseNERE + CHisIEC + BEEVETA + C-CLUE  
**Entity types:** PER · LOC · TITLE · DTM · ORG  
**Backbone:** `ethanyt/guwenbert-base`

---

## Repo Structure

```
├── src/
│   ├── models.py        # GuwenBertCRF, GuwenBertLinear
│   └── data_utils.py    # CoNLL reader, NERDataset
├── api/
│   ├── main.py          # FastAPI app
│   └── predictor.py     # Inference logic
├── outputs/
│   └── ancient/
│       └── guwenbert_crf/
│           ├── best.pt
│           └── label_map.json
├── Dockerfile
└── requirements.txt
```

---

## API

### Start server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t ancient-ner .
docker run -p 8000:8000 ancient-ner
```

### Endpoints

**GET /health**
```json
{"status": "ok", "model_loaded": true}
```

**POST /predict**

Request:
```json
{
  "sentences": "以其徃關上接領北朝頒賞勑諭銀幣故也。\n命胡士楊、阮名實、阮廷正等徃關上候命。"
}
```

Response:
```json
{
  "data": [
    {
      "text": "以其徃關上接領北朝頒賞勑諭銀幣故也。",
      "entities": []
    },
    {
      "text": "命胡士楊、阮名實、阮廷正等徃關上候命。",
      "entities": [
        {"text": "胡士楊", "label": "PER", "start": 1, "end": 4},
        {"text": "阮名實", "label": "PER", "start": 5, "end": 8},
        {"text": "阮廷正", "label": "PER", "start": 9, "end": 12}
      ]
    }
  ]
}
```

### Environment Variables

| Variable    | Default                                          | Description            |
|-------------|--------------------------------------------------|------------------------|
| `BACKBONE`  | `ethanyt/guwenbert-base`                         | HuggingFace model name |
| `CKPT_PATH` | `outputs/ancient/guwenbert_crf/best.pt`          | Model checkpoint       |
| `LABEL_MAP` | `outputs/ancient/guwenbert_crf/label_map.json`   | Label map JSON         |
| `MAX_LEN`   | `128`                                            | Max token length       |

---

## Training

See `notebook45a88c3f07.ipynb` for full training pipeline.
