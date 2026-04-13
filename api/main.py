import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.predictor import AncientNERPredictor


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ancient_ner_api")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


logger = setup_logger()
predictor = None


class PredictRequest(BaseModel):
    sentences: str = Field(..., min_length=1, description="Newline-separated sentences")


class EntityItem(BaseModel):
    type: str
    start: int
    end: int
    text: str


class PredictItem(BaseModel):
    text: str
    entities: List[EntityItem]


class PredictResponse(BaseModel):
    data: List[PredictItem]


def split_sentences(raw_text: str) -> List[str]:
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    return lines


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/best.pt")
    label_map_path = os.getenv("LABEL_MAP_PATH", "artifacts/label_map.json")
    method = os.getenv("MODEL_METHOD", "guwenbert_crf")

    logger.info("Starting API service")
    logger.info("Loading model with method=%s", method)
    logger.info("Checkpoint path: %s", checkpoint_path)
    logger.info("Label map path: %s", label_map_path)

    try:
        predictor = AncientNERPredictor(
            method=method,
            checkpoint_path=checkpoint_path,
            label_map_path=label_map_path,
        )
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        predictor = None

    yield

    logger.info("Shutting down API service")


app = FastAPI(
    title="Ancient Chinese NER API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP error %s at %s: %s", exc.status_code, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error at %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )


@app.get("/health")
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return {"message": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global predictor

    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    if not request.sentences or not request.sentences.strip():
        raise HTTPException(status_code=400, detail="Field 'sentences' must not be empty")

    sentences = split_sentences(request.sentences)

    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found in request body")

    logger.info("Received predict request with %d sentences", len(sentences))

    try:
        results = []
        for sentence in sentences:
            pred = predictor.predict(sentence)
            results.append({
                "text": pred["text"],
                "entities": pred["entities"],
            })

        logger.info("Prediction completed successfully")
        return {"data": results}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed")
