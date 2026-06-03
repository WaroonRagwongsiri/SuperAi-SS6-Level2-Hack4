# main.py
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import (
	OCR_MODEL, LLM_MODEL, THAI_MODEL,
	OCR_PORT, LLM_PORT, THAI_PORT,
	QUEUE_MAX, JOB_TIMEOUT,
)
from worker import ocr_queue, llm_queue, thai_queue, start_workers, TOKEN_STATS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model → (queue, port) routing table ──────────────────────────────────────
# Normalised to lower-case for case-insensitive lookup.
_MODEL_ROUTES: dict[str, tuple[asyncio.Queue, int, str]] = {}


def _build_routes():
	"""Fill routing table after worker queues exist."""
	for model_name, queue, port, stat_key in [
		(OCR_MODEL,  ocr_queue,  OCR_PORT,  "ocr"),
		(LLM_MODEL,  llm_queue,  LLM_PORT,  "llm"),
		(THAI_MODEL, thai_queue, THAI_PORT, "thai"),
	]:
		_MODEL_ROUTES[model_name.lower()] = (queue, port, stat_key)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
	_build_routes()
	start_workers()
	yield


app = FastAPI(
	title="vLLM GPU Switcher",
	description="Unified /v1/chat/completions proxy with GPU sleep/wake management",
	lifespan=lifespan,
)


# ── Core enqueue helper ───────────────────────────────────────────────────────
async def enqueue(queue: asyncio.Queue, payload: dict) -> dict:
	future = asyncio.get_event_loop().create_future()
	try:
		queue.put_nowait((payload, future))
	except asyncio.QueueFull:
		raise HTTPException(status_code=429, detail="Queue full — try again later")
	try:
		return await asyncio.wait_for(future, timeout=JOB_TIMEOUT)
	except asyncio.TimeoutError:
		raise HTTPException(status_code=504, detail="Inference timed out")


# ── /v1/chat/completions ──────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
	"""
	Drop-in replacement for a vLLM /v1/chat/completions endpoint.

	Routes the request to the correct backend based on the `model` field:
	- OCR_MODEL  → port 8003
	- LLM_MODEL  → port 8001
	- THAI_MODEL → port 8002

	The GPU switcher ensures only one model is resident on the GPU at a time.
	The full request body is forwarded as-is so every OpenAI-compatible field
	(messages, temperature, top_p, stream-false, tools, etc.) is preserved.
	"""
	try:
		body: dict[str, Any] = await request.json()
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid JSON body")

	model: str = body.get("model", "")
	if not model:
		raise HTTPException(status_code=400, detail="Missing required field: model")

	route = _MODEL_ROUTES.get(model.lower())
	if route is None:
		available = list(_MODEL_ROUTES.keys())
		raise HTTPException(
			status_code=404,
			detail=f"Model '{model}' not found. Available: {available}",
		)

	queue, _port, stat_key = route
	logger.info(f"[ROUTER] model={model!r} → queue={stat_key.upper()} port={_port}")

	result = await enqueue(queue, body)

	# Patch the model field in the response to echo back what was requested
	# (vLLM may return its internal model name; keep it consistent for clients)
	if isinstance(result, dict) and "model" not in result:
		result["model"] = model

	return JSONResponse(content=result)


# ── /v1/models ────────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
	"""
	Returns the list of available models in the standard OpenAI format.
	Clients (e.g. LiteLLM, Open WebUI) call this to discover what's available.
	"""
	models = [
		{
			"id": OCR_MODEL,
			"object": "model",
			"owned_by": "local",
			"permission": [],
		},
		{
			"id": LLM_MODEL,
			"object": "model",
			"owned_by": "local",
			"permission": [],
		},
		{
			"id": THAI_MODEL,
			"object": "model",
			"owned_by": "local",
			"permission": [],
		},
	]
	return {"object": "list", "data": models}


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
	return {
		"status": "ok",
		"queues": {
			"ocr":  ocr_queue.qsize(),
			"llm":  llm_queue.qsize(),
			"thai": thai_queue.qsize(),
		},
		"cumulative_token_usage": TOKEN_STATS,
	}