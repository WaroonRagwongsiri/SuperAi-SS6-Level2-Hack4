# main.py
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import QUEUE_MAX, JOB_TIMEOUT
from worker import ocr_queue, llm_queue, start_workers, TOKEN_STATS

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


# ── Schemas ───────────────────────────────────────────────────────────────────
class OCRRequest(BaseModel):
	base64_data: str                    
	media_type: str = "image/jpeg"      
	prompt: str = "Extract all text from this document."
	max_tokens: int = 1024

class LLMRequest(BaseModel):
	messages: list                      
	max_tokens: int = 2048
	temperature: float = 0.7


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/ocr")
async def ocr(req: OCRRequest):
	messages = [{
		"role": "user",
		"content": [
			{
				"type": "image_url",
				"image_url": {
					"url": f"data:{req.media_type};base64,{req.base64_data}"
				}
			},
			{"type": "text", "text": req.prompt}
		]
	}]
	return await enqueue(ocr_queue, {"messages": messages, "max_tokens": req.max_tokens})


@app.post("/agent/local")
async def llm(req: LLMRequest):
	return await enqueue(llm_queue, req.model_dump())


@app.get("/health")
async def health():
	return {
		"ocr_queue_size": ocr_queue.qsize(),
		"llm_queue_size": llm_queue.qsize(),
		"cumulative_token_usage": TOKEN_STATS
	}