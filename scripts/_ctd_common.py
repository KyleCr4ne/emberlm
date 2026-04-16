"""Shared glue for the inference HTTP layer.

Supports multiple model families (SmolLM2, Qwen3, LLaMA) by detecting the
family from the safetensors header and loading the matching tokenizer config.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import uuid
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
CTD_GEN = REPO_ROOT / "build" / "bin" / "ctd_generate"

assert CTD_GEN.exists(), f"build ctd_generate first: {CTD_GEN}"


# ---------- model family detection -------------------------------------------

def _read_safetensors_header(path: Path) -> dict:
    """Read only the JSON header of a safetensors file without loading tensor data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def detect_family(weights: Path) -> str:
    """Detect model family from the safetensors header.
    Returns 'smollm2', 'qwen3', or 'llama'."""
    header = _read_safetensors_header(weights)
    vocab = header["model.embed_tokens.weight"]["shape"][0]
    if vocab == 151936:
        return "qwen3"
    if vocab == 49152:
        return "smollm2"
    if vocab == 128256:
        return "llama"
    raise ValueError(f"Unknown model family: vocab_size={vocab} in {weights}")


# ---------- per-family config ------------------------------------------------

def _find_qwen3_tok_dir():
    """Pick whichever Qwen3 dir exists — they share the same tokenizer."""
    for name in ("Qwen3-0.6B", "Qwen3-1.7B"):
        d = REPO_ROOT / "models" / name
        if d.exists():
            return d
    return REPO_ROOT / "models" / "Qwen3-0.6B"  # fallback

_TOKENIZER_DIRS = {
    "smollm2": REPO_ROOT / "models" / "SmolLM2-135M-Instruct",
    "qwen3":   _find_qwen3_tok_dir(),
    "llama":   REPO_ROOT / "models" / "Llama-3.2-1B-Instruct",
}

_EOS_IDS = {
    "smollm2": [0, 2],       # <|endoftext|>=0, <|im_end|>=2
    "qwen3":   [151645, 151643],  # <|im_end|>=151645, <|endoftext|>=151643
    "llama":   [128001, 128008, 128009],  # <|end_of_text|>, <|eot_id|>, <|eom_id|>
}

# SmolLM2's BPE splits <think>/<think> into 3 tokens each;
# Qwen3 has them as single special tokens.
_THINK_OPEN = {
    "smollm2": [44, 17400, 46],     # "<", "think", ">"
    "qwen3":   [151667],             # <think>
    "llama":   [],
}
_THINK_CLOSE = {
    "smollm2": [9617, 17400, 46],    # "</", "think", ">"
    "qwen3":   [151668],             # </think>
    "llama":   [],
}


class ModelConfig:
    """Holds the tokenizer + family-specific constants for one model family."""

    def __init__(self, family: str):
        self.family = family
        tok_dir = _TOKENIZER_DIRS[family]
        assert tok_dir.exists(), f"tokenizer dir missing: {tok_dir} — run fetch_model.py"
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        self.eos_ids = _EOS_IDS[family]
        self.think_open_ids = _THINK_OPEN[family]
        self.think_close_ids = _THINK_CLOSE[family]

    def render_prompt(self, messages: list[dict]) -> list[int]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return ids


@lru_cache(maxsize=4)
def get_model_config(weights: Path) -> ModelConfig:
    family = detect_family(weights)
    return ModelConfig(family)


# ---------- subprocess registry ----------------------------------------------
_REGISTRY: dict[str, list[asyncio.subprocess.Process]] = {}


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def register_proc(req_id: str, proc: asyncio.subprocess.Process) -> None:
    _REGISTRY.setdefault(req_id, []).append(proc)


def unregister_all(req_id: str) -> None:
    _REGISTRY.pop(req_id, None)


async def stop_request(req_id: str) -> int:
    procs = _REGISTRY.pop(req_id, [])
    n = 0
    for p in procs:
        if p.returncode is None:
            p.terminate()
            n += 1
    return n


# ---------- generation params ------------------------------------------------
class GenParams(dict):
    @classmethod
    def from_request(cls, r: dict) -> "GenParams":
        return cls(
            max_new_tokens=int(r.get("max_new_tokens", 256)),
            temperature=float(r.get("temperature", 0.7)),
            top_k=int(r.get("top_k", 50)),
            top_p=float(r.get("top_p", 0.9)),
            seed=int(r.get("seed", 0)),
            thinking_token_budget=int(r.get("thinking_token_budget", 0)),
        )


async def stream_generate(
    weights: Path,
    input_ids: list[int],
    params: GenParams,
    req_id: str,
    *,
    model_cfg: ModelConfig | None = None,
    max_seq_len: int = 4096,
) -> AsyncIterator[dict]:
    """Spawn one ctd_generate, yield parsed JSON events."""
    if model_cfg is None:
        model_cfg = get_model_config(weights)

    payload = {
        "input_ids": input_ids,
        "max_new_tokens": params["max_new_tokens"],
        "temperature": params["temperature"],
        "top_k": params["top_k"],
        "top_p": params["top_p"],
        "seed": params["seed"],
        "eos_ids": model_cfg.eos_ids,
        "max_seq_len": max_seq_len,
        "thinking_token_budget": params["thinking_token_budget"],
        "think_open_ids": model_cfg.think_open_ids,
        "think_close_ids": model_cfg.think_close_ids,
    }

    proc = await asyncio.create_subprocess_exec(
        str(CTD_GEN),
        "--model", str(weights),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    register_proc(req_id, proc)
    proc.stdin.write(json.dumps(payload).encode() + b"\n")
    await proc.stdin.drain()
    proc.stdin.close()

    tokenizer = model_cfg.tokenizer
    eos_set = set(model_cfg.eos_ids)
    generated: list[int] = []
    decoded_so_far = ""
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "token" in evt:
                tok = int(evt["token"])
                if tok in eos_set:
                    continue
                generated.append(tok)
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if len(text) > len(decoded_so_far):
                    delta = text[len(decoded_so_far):]
                    decoded_so_far = text
                    yield {"delta": delta}
            elif evt.get("done"):
                yield {"done": True, "reason": evt.get("reason")}
                break
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
