# worker.py
import asyncio
import logging
import httpx
from config import *

logger = logging.getLogger(__name__)

# ── Global Token Tracking ───────────────────────────────────────────────────
TOKEN_STATS = {
	"ocr":  {"prompt": 0, "completion": 0, "total": 0},
	"llm":  {"prompt": 0, "completion": 0, "total": 0},
	"thai": {"prompt": 0, "completion": 0, "total": 0},
}

# ── GPU state tracking ───────────────────────────────────────────────────────
_awake_port: int | None = None

# ── One global lock — only 1 model awake at a time ──────────────────────────
GPU_LOCK = asyncio.Lock()

# ── 3 independent queues ─────────────────────────────────────────────────────
ocr_queue  = asyncio.Queue(maxsize=QUEUE_MAX)
llm_queue  = asyncio.Queue(maxsize=QUEUE_MAX)
thai_queue = asyncio.Queue(maxsize=QUEUE_MAX)

http = httpx.AsyncClient(timeout=300.0)

# ── Wake / Sleep helpers ──────────────────────────────────────────────────────
async def sleep_port(port: int):
	logger.info(f"[:{port}] SLEEP → sending sleep?level=2 ...")
	r = await http.post(f"{BASE_URL}:{port}/sleep?level=2")
	r.raise_for_status()
	logger.info(f"[:{port}] SLEEP → done (status {r.status_code})")


async def wake_port(port: int):
	"""
	Full 3-step wake sequence with explicit waits between steps:
	1. wake_up?tags=weights    — reallocate weights memory
	2. collective_rpc          — reload weights in-place onto GPU
	3. wake_up?tags=kv_cache   — reallocate KV cache on GPU
	Each step is confirmed before proceeding to the next.
	"""
	logger.info(f"[:{port}] WAKE STEP 1/3 → wake_up?tags=weights ...")
	r = await http.post(f"{BASE_URL}:{port}/wake_up?tags=weights")
	r.raise_for_status()
	logger.info(f"[:{port}] WAKE STEP 1/3 → weights memory allocated (status {r.status_code})")

	# Small buffer — give vLLM time to finish allocating before we push weights
	await asyncio.sleep(0.5)

	logger.info(f"[:{port}] WAKE STEP 2/3 → collective_rpc reload_weights ...")
	r = await http.post(
		f"{BASE_URL}:{port}/collective_rpc",
		json={"method": "reload_weights"},
	)
	r.raise_for_status()
	logger.info(f"[:{port}] WAKE STEP 2/3 → weights loaded in-place (status {r.status_code})")

	# Wait for weights to be fully mapped to GPU before allocating KV cache
	await asyncio.sleep(0.5)

	logger.info(f"[:{port}] WAKE STEP 3/3 → wake_up?tags=kv_cache ...")
	r = await http.post(f"{BASE_URL}:{port}/wake_up?tags=kv_cache")
	r.raise_for_status()
	logger.info(f"[:{port}] WAKE STEP 3/3 → KV cache on GPU (status {r.status_code})")

	# Final buffer before we allow inference — ensures KV cache is fully live
	await asyncio.sleep(0.3)

	logger.info(f"[:{port}] WAKE → fully ready for inference")


async def ensure_awake(port: int):
	global _awake_port

	if _awake_port == port:
		logger.info(f"[:{port}] already awake — skipping wake sequence")
		return

	if _awake_port is not None:
		logger.info(f"[:{_awake_port}] preempted by :{port} — sleeping first")
		await sleep_port(_awake_port)
		_awake_port = None
		# Buffer after sleep before waking next model
		await asyncio.sleep(0.5)

	await wake_port(port)
	_awake_port = port


# ── Model calls ───────────────────────────────────────────────────────────────
async def call_ocr(payload: dict) -> dict:
	logger.info(f"[OCR] → sending request to :{OCR_PORT}")
	r = await http.post(
		f"{BASE_URL}:{OCR_PORT}/v1/chat/completions",
		json={
			"model": OCR_MODEL,
			"messages": payload["messages"],
			"max_tokens": payload.get("max_tokens", 10240),
		},
	)
	r.raise_for_status()
	logger.info(f"[OCR] → response received (status {r.status_code})")
	return r.json()


async def call_llm(payload: dict) -> dict:
	logger.info(f"[LLM] → sending request to :{LLM_PORT}")
	r = await http.post(
		f"{BASE_URL}:{LLM_PORT}/v1/chat/completions",
		json={
			"model": LLM_MODEL,
			"messages": payload["messages"],
			"max_tokens": payload.get("max_tokens", 8196),
			"temperature": payload.get("temperature", 0.2),
		},
	)
	r.raise_for_status()
	logger.info(f"[LLM] → response received (status {r.status_code})")
	return r.json()


async def call_thai(payload: dict) -> dict:
	logger.info(f"[THAI] → sending request to :{THAI_PORT}")
	r = await http.post(
		f"{BASE_URL}:{THAI_PORT}/v1/chat/completions",
		json={
			"model": THAI_MODEL,
			"messages": payload["messages"],
			"max_tokens": payload.get("max_tokens", 8196),
			"temperature": payload.get("temperature", 0.2),
		},
	)
	r.raise_for_status()
	logger.info(f"[THAI] → response received (status {r.status_code})")
	return r.json()


# ── Generic worker loop ───────────────────────────────────────────────────────
async def worker_loop(name: str, port: int, queue: asyncio.Queue, call_fn):
	logger.info(f"[{name}] worker ready, listening on queue")
	key = name.lower()

	while True:
		logger.info(f"[{name}] waiting for job ...")
		payload, future = await queue.get()
		logger.info(f"[{name}] job dequeued — acquiring GPU_LOCK ...")

		try:
			async with GPU_LOCK:
				logger.info(f"[{name}] GPU_LOCK acquired")

				await ensure_awake(port)

				logger.info(f"[{name}] starting inference ...")
				result = await call_fn(payload)
				logger.info(f"[{name}] inference finished")

				# ── Token tracking ───────────────────────────────────────────
				usage = result.get("usage", {})
				p_tok = usage.get("prompt_tokens", 0)
				c_tok = usage.get("completion_tokens", 0)
				t_tok = usage.get("total_tokens", 0)

				TOKEN_STATS[key]["prompt"]     += p_tok
				TOKEN_STATS[key]["completion"] += c_tok
				TOKEN_STATS[key]["total"]      += t_tok

				logger.info(
					f"[{name}] tokens → this run: {t_tok} "
					f"(prompt: {p_tok}, output: {c_tok}) | "
					f"cumulative: {TOKEN_STATS[key]['total']}"
				)

				future.set_result(result)
				logger.info(f"[{name}] future resolved — GPU_LOCK releasing")

		except Exception as e:
			logger.error(f"[{name}] FAILED: {e}", exc_info=True)
			if not future.done():
				future.set_exception(e)
		finally:
			queue.task_done()
			logger.info(f"[{name}] task_done — back to waiting")


# ── Start all workers (call once at startup) ──────────────────────────────────
def start_workers():
	asyncio.create_task(worker_loop("OCR",  OCR_PORT,  ocr_queue,  call_ocr))
	asyncio.create_task(worker_loop("LLM",  LLM_PORT,  llm_queue,  call_llm))
	asyncio.create_task(worker_loop("THAI", THAI_PORT, thai_queue, call_thai))