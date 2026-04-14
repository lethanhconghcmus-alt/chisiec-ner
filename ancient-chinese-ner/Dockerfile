FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code only — checkpoints mounted via docker-compose
COPY src/  src/
COPY api/  api/

# Env defaults (override at runtime or via docker-compose)
ENV BACKBONE=ethanyt/guwenbert-base
ENV CKPT_PATH=outputs/ancient/guwenbert_crf/best.pt
ENV LABEL_MAP=outputs/ancient/guwenbert_crf/label_map.json
ENV MAX_LEN=128

# Recommended: --memory=3g in docker-compose (model ~1.2GB, peak ~1.5GB)

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
