"""Inference-only HTTP server.

Usage:
    python scripts/serve_inference.py --solo runs/<name>/final.safetensors
    python scripts/serve_inference.py --sbs runs/a/final.safetensors runs/b/final.safetensors

Solo mode serves one model; SBS mode serves two side-by-side for visual A/B
comparison. The UI discovers the mode via GET /api/mode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from _ctd_common import (
    GenParams,
    REPO_ROOT,
    get_model_config,
    new_request_id,
    stop_request,
    stream_generate,
    unregister_all,
)

WEB_DIR = REPO_ROOT / "web"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--solo", metavar="WEIGHTS", help="path to a single safetensors file")
    g.add_argument(
        "--sbs",
        nargs=2,
        metavar=("WEIGHTS_A", "WEIGHTS_B"),
        help="two weights files for side-by-side comparison",
    )
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    return p.parse_args()


ARGS = parse_args()

if ARGS.solo:
    WEIGHTS = {"solo": Path(ARGS.solo)}
    MODE = "solo"
else:
    WEIGHTS = {"a": Path(ARGS.sbs[0]), "b": Path(ARGS.sbs[1])}
    MODE = "sbs"

for slot, p in WEIGHTS.items():
    assert p.exists(), f"weights for slot {slot!r} missing: {p}"

MODEL_CFGS = {slot: get_model_config(w) for slot, w in WEIGHTS.items()}

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class SoloChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    seed: int = 0
    thinking_token_budget: int = 0


class SBSChatRequest(BaseModel):
    messages: list[ChatMessage]
    slot_a: dict
    slot_b: dict


@app.get("/api/mode")
def api_mode():
    """Tell the UI whether to render one or two columns, and what labels."""
    if MODE == "solo":
        return {"mode": "solo", "slots": [{"id": "solo", "label": WEIGHTS["solo"].name}]}
    return {
        "mode": "sbs",
        "slots": [
            {"id": "a", "label": WEIGHTS["a"].parent.name + "/" + WEIGHTS["a"].name},
            {"id": "b", "label": WEIGHTS["b"].parent.name + "/" + WEIGHTS["b"].name},
        ],
    }


async def _sse_event(obj: dict) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


async def _run_solo(req: SoloChatRequest, req_id: str):
    mcfg = MODEL_CFGS["solo"]
    try:
        ids = mcfg.render_prompt([m.model_dump() for m in req.messages])
        params = GenParams.from_request(req.model_dump())
        yield await _sse_event({"req_id": req_id, "slot": "solo", "start": True})
        async for evt in stream_generate(WEIGHTS["solo"], ids, params, req_id, model_cfg=mcfg):
            evt["slot"] = "solo"
            yield await _sse_event(evt)
    finally:
        unregister_all(req_id)


async def _run_sbs(req: SBSChatRequest, req_id: str):
    """Fan out to two subprocesses, interleave their SSE events tagged by slot.

    Each slot gets its own tokenizer/EOS config. Slots can be different model
    families (e.g. SmolLM2 vs Qwen3) — each prompt is rendered with its own
    tokenizer.
    """
    mcfg_a = MODEL_CFGS["a"]
    mcfg_b = MODEL_CFGS["b"]
    ids_a = mcfg_a.render_prompt([m.model_dump() for m in req.messages])
    ids_b = mcfg_b.render_prompt([m.model_dump() for m in req.messages])
    params_a = GenParams.from_request(req.slot_a)
    params_b = GenParams.from_request(req.slot_b)

    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    async def pump(slot: str, weights: Path, params: GenParams, ids: list[int],
                   mcfg=None):
        try:
            async for evt in stream_generate(weights, ids, params, req_id, model_cfg=mcfg):
                evt["slot"] = slot
                await queue.put(evt)
        except Exception as e:
            await queue.put({"slot": slot, "error": str(e)})
        finally:
            await queue.put(SENTINEL)

    task_a = asyncio.create_task(pump("a", WEIGHTS["a"], params_a, ids_a, mcfg_a))
    task_b = asyncio.create_task(pump("b", WEIGHTS["b"], params_b, ids_b, mcfg_b))

    try:
        yield await _sse_event({"req_id": req_id, "start": True})
        remaining = 2
        while remaining > 0:
            evt = await queue.get()
            if evt is SENTINEL:
                remaining -= 1
                continue
            yield await _sse_event(evt)
        yield await _sse_event({"done_all": True})
    finally:
        for t in (task_a, task_b):
            if not t.done():
                t.cancel()
        unregister_all(req_id)


@app.post("/chat")
async def chat(req: SoloChatRequest):
    if MODE != "solo":
        return JSONResponse({"error": "server is running in sbs mode"}, status_code=400)
    req_id = new_request_id()
    return StreamingResponse(
        _run_solo(req, req_id),
        media_type="text/event-stream",
        headers={"X-Request-Id": req_id},
    )


@app.post("/chat/sbs")
async def chat_sbs(req: SBSChatRequest):
    if MODE != "sbs":
        return JSONResponse({"error": "server is running in solo mode"}, status_code=400)
    req_id = new_request_id()
    return StreamingResponse(
        _run_sbs(req, req_id),
        media_type="text/event-stream",
        headers={"X-Request-Id": req_id},
    )


@app.post("/stop/{req_id}")
async def stop(req_id: str):
    n = await stop_request(req_id)
    return {"stopped": n}


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    print(f"[serve_inference] mode={MODE} weights={dict((k, str(v)) for k, v in WEIGHTS.items())}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")
