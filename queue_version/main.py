# main.py
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import QUEUE_MAX, JOB_TIMEOUT
from worker import ocr_queue, llm_queue, thai_queue, start_workers, TOKEN_STATS

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
	start_workers()
	yield


app = FastAPI(lifespan=lifespan)


# ── Helper ────────────────────────────────────────────────────────────────────
async def enqueue(queue: asyncio.Queue, payload: dict) -> dict:
	future = asyncio.get_event_loop().create_future()
	try:
		queue.put_nowait((payload, future))
	except asyncio.QueueFull:
		raise HTTPException(status_code=429, detail="Queue full")
	return await asyncio.wait_for(future, timeout=JOB_TIMEOUT)


def _extract_answer(raw: dict) -> str:
	"""Pull the assistant text out of a /v1/chat/completions response."""
	try:
		return raw["choices"][0]["message"]["content"]
	except (KeyError, IndexError):
		return ""


def _extract_json_answer(raw: dict) -> dict:
	"""
	OCR: the model is asked to return JSON.
	Try to parse the content as JSON; fall back to returning it as-is under 'raw'.
	"""
	import json
	text = _extract_answer(raw)
	try:
		return json.loads(text)
	except (json.JSONDecodeError, TypeError):
		return {"raw": text}


# ── Schemas ───────────────────────────────────────────────────────────────────
class OCRRequest(BaseModel):
	image: dict  # {"header": "<base64>", "transaction": "<base64>"}


class AgentRequest(BaseModel):
	question: str
	max_tokens: int = 4096
	temperature: float = 0.2


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ocr")
async def ocr(req: OCRRequest):
	"""
	Input:
		{ "image": { "header": "<base64>", "transaction": "<base64>" } }
	Output:
		JSON object parsed from the model's response
	"""
	header_b64      = req.image.get("header", "")
	transaction_b64 = req.image.get("transaction", "")

	content = []
	if header_b64:
		content.append({
			"type": "image_url",
			"image_url": {"url": f"data:image/jpeg;base64,{header_b64}"}
		})
	if transaction_b64:
		content.append({
			"type": "image_url",
			"image_url": {"url": f"data:image/jpeg;base64,{transaction_b64}"}
		})
	content.append({
		"type": "text",
		"text": (
			"Extract all text from both images and return ONLY a valid JSON object. "
			"Use keys 'header' and 'transaction' for each image's extracted text."
		)
	})

	messages = [{"role": "user", "content": content}]
	raw = await enqueue(ocr_queue, {"messages": messages, "max_tokens": 1024})
	return _extract_json_answer(raw)


@app.post("/agent/local")
async def agent_local(req: AgentRequest):
	"""
	Gemma LLM endpoint.
	Input:  { "question": "..." }
	Output: { "answer": "...", "total_output_token": N }
	"""
	messages = [{"role": "user", "content": req.question}]
	raw = await enqueue(
		llm_queue,
		{"messages": messages, "max_tokens": req.max_tokens, "temperature": req.temperature},
	)
	usage = raw.get("usage", {})
	return {
		"answer": _extract_answer(raw),
		"total_output_token": TOKEN_STATS["llm"]["total"],
	}


@app.post("/agent/thaillm")
async def agent_thaillm(req: AgentRequest):
	"""
	Thai LLM endpoint.
	Input:  { "question": "..." }
	Output: { "answer": "...", "total_output_token": N }
	"""
	messages = [{"role": "user", "content": req.question}]
	raw = await enqueue(
		thai_queue,
		{"messages": messages, "max_tokens": req.max_tokens, "temperature": req.temperature},
	)
	return {
		"answer": _extract_answer(raw),
		"total_output_token": TOKEN_STATS["thai"]["total"],
	}


@app.get("/health")
async def health():
	return {
		"ocr_queue_size":  ocr_queue.qsize(),
		"llm_queue_size":  llm_queue.qsize(),
		"thai_queue_size": thai_queue.qsize(),
		"cumulative_token_usage": TOKEN_STATS,
	}
