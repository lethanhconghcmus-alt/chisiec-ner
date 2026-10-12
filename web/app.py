"""
Web local: tách câu + NER Hán văn cổ, so sánh nhiều model.

  python3 -m uvicorn web.app:app --port 8000     (chạy từ thư mục gốc repo)
  → mở http://localhost:8000
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.evaluate import evaluate, load_gold
from web.pipeline import DEVICE, NER_MODELS, SEGMENTERS, Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
STATIC = Path(__file__).parent / "static"
MAX_INPUT = 20000

pipeline = Pipeline()  # model nạp khi dùng lần đầu

app = FastAPI(title="Hán cổ — tách câu + NER")


class AnalyzeRequest(BaseModel):
    text: str
    segmenter: str = "guwen-punc"
    models: List[str] = ["gw-crf-tq", "guwen-ner"]
    simplify: bool = False


class EvaluateRequest(BaseModel):
    segmenter: str = "guwen-punc"
    models: List[str] = ["gw-crf-tq", "guwen-ner"]
    simplify: bool = False


def _check(segmenter: str, models: List[str]) -> None:
    if segmenter not in SEGMENTERS:
        raise HTTPException(400, f"Không có model ngắt câu '{segmenter}'")
    bad = [m for m in models if m not in NER_MODELS]
    if bad or not models:
        raise HTTPException(400, f"Model NER không hợp lệ: {bad or 'chưa chọn model'}")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/gold")
def gold():
    return FileResponse(STATIC / "gold.html")


@app.get("/api/gold")
def gold_manual():
    return {"docs": load_gold()}


@app.get("/api/info")
def info():
    return {
        "device": str(DEVICE),
        "segmenters": [{"id": k, "name": v["name"], "desc": v["desc"]} for k, v in SEGMENTERS.items()],
        "models": [{"id": k, "name": v["name"], "desc": v["desc"]} for k, v in NER_MODELS.items()],
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    _check(req.segmenter, req.models)
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Chưa nhập văn bản")
    if len(text) > MAX_INPUT:
        raise HTTPException(413, f"Văn bản quá dài (>{MAX_INPUT} chữ)")
    return pipeline.run(text, req.segmenter, req.models, req.simplify)


@app.post("/api/evaluate")
def evaluate_gold(req: EvaluateRequest):
    _check(req.segmenter, req.models)
    return evaluate(pipeline, req.segmenter, req.models, req.simplify)
