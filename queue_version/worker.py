# worker.py
import asyncio
import logging
import httpx
from config import *

logger = logging.getLogger(__name__)

# ── Global Token Tracking ───────────────────────────────────────────────────
TOKEN_STATS = {
	"ocr": {"prompt": 0, "completion": 0, "total": 0},
	"llm": {"prompt": 0, "completion": 0, "total": 0},
}

# ── One global lock — only 1 model awake at a time ──────────────────────────
GPU_LOCK = asyncio.Lock()

# ── 2 independent queues ─────────────────────────────────────────────────────
ocr_queue = asyncio.Queue(maxsize=QUEUE_MAX)
llm_queue = asyncio.Queue(maxsize=QUEUE_MAX)

http = httpx.AsyncClient(timeout=300.0)


# ── Wake / Sleep helpers ──────────────────────────────────────────────────────
async def wake(port: int):
	await http.post(f"{BASE_URL}:{port}/wake_up")
	logger.info(f"[:{port}] awake")

async def sleep(port: int):
	await http.post(f"{BASE_URL}:{port}/sleep")
	logger.info(f"[:{port}] sleeping")


# ── Model calls ───────────────────────────────────────────────────────────────
async def call_ocr(payload: dict) -> dict:
	r = await http.post(
		f"{BASE_URL}:{OCR_PORT}/v1/chat/completions",
		json={
			"model": OCR_MODEL,
			"messages": payload["messages"],
			"max_tokens": payload.get("max_tokens", 1024),
		},
	)
	r.raise_for_status()
	return r.json()


async def call_llm(payload: dict) -> dict:
	r = await http.post(
		f"{BASE_URL}:{LLM_PORT}/v1/chat/completions",
		json={
			"model": LLM_MODEL,
			"messages": payload["messages"],
			"max_tokens": payload.get("max_tokens", 2048),
			"temperature": payload.get("temperature", 0.7),
		},
	)
	r.raise_for_status()
	return r.json()


# ── Generic worker loop ───────────────────────────────────────────────────────
async def worker_loop(name: str, port: int, queue: asyncio.Queue, call_fn):
	logger.info(f"[{name}] worker ready")
	key = name.lower()  # "ocr" or "llm"
	
	while True:
		payload, future = await queue.get() # blocks until a job arrives
		try:
			async with GPU_LOCK: # wait until GPU is free
				await wake(port)
				try:
					result = await call_fn(payload)

					# ── Extract and Track Token Usage ────────────────────────
					usage = result.get("usage", {})
					p_tok = usage.get("prompt_tokens", 0)
					c_tok = usage.get("completion_tokens", 0)
					t_tok = usage.get("total_tokens", 0)

					# Update global cumulative stats
					TOKEN_STATS[key]["prompt"] += p_tok
					TOKEN_STATS[key]["completion"] += c_tok
					TOKEN_STATS[key]["total"] += t_tok

					logger.info(
						f"[{name}] Inference complete. "
						f"Tokens used -> This run: {t_tok} (Prompt: {p_tok}, Output: {c_tok}) | "
						f"Cumulative Total: {TOKEN_STATS[key]['total']}"
					)
					future.set_result(result)
				finally:
					await sleep(port)
		except Exception as e:
			logger.error(f"[{name}] failed: {e}")
			if not future.done():
				future.set_exception(e)
		finally:
			queue.task_done()


# ── Start all workers (call once at startup) ──────────────────────────────────
def start_workers():
	asyncio.create_task(worker_loop("OCR", OCR_PORT, ocr_queue, call_ocr))
	asyncio.create_task(worker_loop("LLM", LLM_PORT, llm_queue, call_llm))