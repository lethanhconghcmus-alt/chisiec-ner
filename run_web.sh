#!/usr/bin/env bash
# Chạy web local tách câu + NER: ./run_web.sh  → http://localhost:8000
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
exec python3 -m uvicorn web.app:app --host 127.0.0.1 --port "$PORT"
