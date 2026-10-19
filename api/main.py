"""
FastAPI backend for Ancient Chinese sentence segmentation + NER.

Endpoints:
  GET  /health   — liveness check
  GET  /models   — available NER models
  POST /predict  — NER inference (one sentence per line)
  POST /segment  — sentence segmentation (断句)
  POST /analyze  — sentence segmentation + NER

/predict and /analyze take an optional "model" (see GET /models); default = NER_MODEL.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from api.predictor import (DEFAULT_MODEL, NER_MODELS, Predictor, available_models,
                           get_predictor, loaded_models)
from api.segmenter import Segmenter, split_sentences

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api.main")

MAX_INPUT = int(os.getenv("MAX_INPUT", "20000"))  # max chars for /segment and /analyze

# ── Global models (default NER model + segmenter loaded at startup) ─
segmenter: Segmenter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global segmenter
    logger.info("Startup: loading models … (available NER models: %s)", available_models())
    get_predictor(DEFAULT_MODEL)
    segmenter = Segmenter()
    logger.info("Startup: models ready")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Ancient Chinese Segmentation + NER API",
    version="1.2.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sentences: str
    model: Optional[str] = None

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
    model: str
    data:  List[SentenceOut]


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
    model: Optional[str] = None


class SegmentResponse(BaseModel):
    punctuated: str
    sentences:  List[str]


class AnalyzeResponse(BaseModel):
    model:      str
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


def _predictor(name: Optional[str]) -> Predictor:
    try:
        return get_predictor(name)
    except KeyError:
        raise HTTPException(status_code=400,
                            detail=f"Unknown or unavailable model '{name}'. Available: {available_models()}")


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": DEFAULT_MODEL in loaded_models(),
        "segmenter_loaded": segmenter is not None,
        "default_model": DEFAULT_MODEL,
        "loaded_models": loaded_models(),
    }


@app.get("/models", tags=["NER"])
def models():
    return {
        "default": DEFAULT_MODEL,
        "models": [{"name": n, "backbone": NER_MODELS[n]["backbone"],
                    "max_len": NER_MODELS[n]["max_len"], "description": NER_MODELS[n]["description"],
                    "loaded": n in loaded_models()} for n in available_models()],
    }


@app.post("/predict", response_model=PredictResponse, tags=["NER"])
def predict(req: PredictRequest):
    predictor = _predictor(req.model)

    # Split on newline — each line is one sentence
    raw_sentences = [s for s in req.sentences.split("\n") if s.strip()]
    if not raw_sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")

    logger.info("Predicting %d sentence(s) | model=%s", len(raw_sentences), predictor.name)

    results: List[SentenceOut] = []
    for sent in raw_sentences:
        try:
            # window long lines instead of truncating them at max_len
            entities = predictor.predict_document([sent])[0]
            results.append(SentenceOut(text=sent, entities=entities))
        except Exception as exc:
            logger.exception("Prediction failed for sentence: %s | %s", sent[:40], exc)
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictResponse(model=predictor.name, data=results)


@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"])
def segment(req: TextRequest):
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    sentences = segmenter.segment(req.text)
    logger.info("Segmented %d char(s) into %d sentence(s)", len(req.text), len(sentences))
    return SegmentResponse(punctuated="".join(sentences), sentences=sentences)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["NER"])
def analyze(req: AnalyzeRequest):
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    predictor = _predictor(req.model)

    if req.auto_segment:
        # Drop existing punctuation and let the model re-segment
        sentences = segmenter.segment(req.text)
    else:
        # Keep the input as is: split on newlines and sentence-final marks
        sentences = [s for ln in req.text.splitlines() for s in split_sentences(re.sub(r"\s", "", ln))]
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")

    logger.info("Analyzing %d sentence(s) | auto_segment=%s | model=%s",
                len(sentences), req.auto_segment, predictor.name)
    entities = predictor.predict_document(sentences)
    return AnalyzeResponse(
        model=predictor.name,
        punctuated="".join(sentences),
        data=[SentenceOut(text=s, entities=e) for s, e in zip(sentences, entities)],
    )
