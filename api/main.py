"""
FastAPI backend for Ancient Chinese NER.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — NER inference
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from api.predictor import Predictor

# ── Logging setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api.main")

# ── Global predictor (loaded once at startup) ──────────────────────
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Startup: loading model …")
    predictor = Predictor()
    logger.info("Startup: model ready")
    yield
    logger.info("Shutdown")


app = FastAPI(
    title="Ancient Chinese NER API",
    version="1.0.0",
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
    return {"status": "ok", "model_loaded": predictor is not None}


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
