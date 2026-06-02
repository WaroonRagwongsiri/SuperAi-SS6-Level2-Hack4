#!/bin/bash
# /root/vllm/start_gateway.sh

source /root/vllm/.venv/bin/activate

uv pip install fastapi uvicorn httpx pydantic-settings

# Sleep both models (ignore errors if already sleeping)
curl -s -X POST http://localhost:8000/sleep > /dev/null || true
curl -s -X POST http://localhost:8001/sleep > /dev/null || true

# Bind to 0.0.0.0 so it's accessible externally, and use exec
exec uvicorn main:app --host 0.0.0.0 --port 8080
