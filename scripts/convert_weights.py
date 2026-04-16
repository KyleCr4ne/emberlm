"""Convert model bf16/fp16 safetensors to a single fp32 safetensors file.

Multi-shard models are merged into one `model.safetensors` so the C++ loader
can read a single file.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M-Instruct"
SRC_DIR = REPO_ROOT / "models" / MODEL_DIR_NAME
DST_DIR = SRC_DIR / "fp32"


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    src_files = sorted(SRC_DIR.glob("*.safetensors"))
    src_files = [p for p in src_files if "fp32" not in p.parts]
    if not src_files:
        raise SystemExit(f"no .safetensors under {SRC_DIR}; run fetch_model.py first")

    merged: dict[str, torch.Tensor] = {}
    for src in src_files:
        print(f"[convert] reading {src.name}")
        with safe_open(src, framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                merged[key] = t.to(torch.float32).contiguous()

    dst = DST_DIR / "model.safetensors"
    print(f"[convert] writing {len(merged)} tensors -> {dst}")
    save_file(merged, str(dst))
    print(f"[convert]   {dst.stat().st_size / 1024**2:.2f} MiB")

    for old in DST_DIR.glob("model-*-of-*.safetensors"):
        old.unlink()
        print(f"[convert]   removed stale shard {old.name}")


if __name__ == "__main__":
    main()
