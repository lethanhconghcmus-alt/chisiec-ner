"""
FastAPI backend for Ancient Chinese sentence segmentation + NER.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — NER inference (one sentence per line)
  POST /segment  — sentence segmentation (断句)
  POST /analyze  — sentence segmentation + NER
"""
from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from api.predictor import Predictor
from api.segmenter import Segmenter, split_sentences

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api.main")

MAX_INPUT = int(os.getenv("MAX_INPUT", "20000"))  # max chars for /segment and /analyze

# ── Global models (loaded once at startup) ─────────────────────────
predictor: Predictor | None = None
segmenter: Segmenter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, segmenter
    logger.info("Startup: loading models …")
    predictor = Predictor()
    segmenter = Segmenter()
    logger.info("Startup: models ready")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Ancient Chinese Segmentation + NER API",
    version="1.1.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sentences: str

    @field_validator("sentences")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("sentences must not be empty")
        return v


class EntityOut(BaseModel):
    text:  str
    label: str
    start: int
    end:   int


class SentenceOut(BaseModel):
    text:     str
    entities: List[EntityOut]


class PredictResponse(BaseModel):
    data: List[SentenceOut]


class TextRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > MAX_INPUT:
            raise ValueError(f"text must not exceed {MAX_INPUT} characters")
        return v


class AnalyzeRequest(TextRequest):
    auto_segment: bool = True


class SegmentResponse(BaseModel):
    punctuated: str
    sentences:  List[str]


class AnalyzeResponse(BaseModel):
    punctuated: str
    data:       List[SentenceOut]


# ── Exception handlers ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "segmenter_loaded": segmenter is not None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["NER"])
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Split on newline — each line is one sentence
    raw_sentences = [s for s in req.sentences.split("\n") if s.strip()]
    if not raw_sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")

    logger.info("Predicting %d sentence(s)", len(raw_sentences))

    results: List[SentenceOut] = []
    for sent in raw_sentences:
        try:
            entities = predictor.predict(sent)
            results.append(SentenceOut(text=sent, entities=entities))
        except Exception as exc:
            logger.exception("Prediction failed for sentence: %s | %s", sent[:40], exc)
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictResponse(data=results)


@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"])
def segment(req: TextRequest):
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    sentences = segmenter.segment(req.text)
    logger.info("Segmented %d char(s) into %d sentence(s)", len(req.text), len(sentences))
    return SegmentResponse(punctuated="".join(sentences), sentences=sentences)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["NER"])
def analyze(req: AnalyzeRequest):
    if predictor is None or segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if req.auto_segment:
        # Drop existing punctuation and let the model re-segment
        sentences = segmenter.segment(req.text)
    else:
        # Keep the input as is: split on newlines and sentence-final marks
        sentences = [s for ln in req.text.splitlines() for s in split_sentences(re.sub(r"\s", "", ln))]
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")

    logger.info("Analyzing %d sentence(s) | auto_segment=%s", len(sentences), req.auto_segment)
    entities = predictor.predict_document(sentences)
    return AnalyzeResponse(
        punctuated="".join(sentences),
        data=[SentenceOut(text=s, entities=e) for s, e in zip(sentences, entities)],
    )
