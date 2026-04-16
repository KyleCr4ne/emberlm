"""Download model weights from Hugging Face.

Supported families:
    SmolLM2:  --model smollm2 --size 135M|360M [--base]
    Qwen3:    --model qwen3   --size 0.6B
    LLaMA:    --model llama   --size 1B [--base]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    ("smollm2", "135M", "instruct"): ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct"),
    ("smollm2", "135M", "base"):     ("HuggingFaceTB/SmolLM2-135M",          "SmolLM2-135M"),
    ("smollm2", "360M", "instruct"): ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-360M-Instruct"),
    ("smollm2", "360M", "base"):     ("HuggingFaceTB/SmolLM2-360M",          "SmolLM2-360M"),
    ("qwen3",   "0.6B", "instruct"): ("Qwen/Qwen3-0.6B",                     "Qwen3-0.6B"),
    ("qwen3",   "1.7B", "instruct"): ("Qwen/Qwen3-1.7B",                     "Qwen3-1.7B"),
    ("llama",   "1B",   "instruct"): ("meta-llama/Llama-3.2-1B-Instruct",   "Llama-3.2-1B-Instruct"),
    ("llama",   "1B",   "base"):     ("meta-llama/Llama-3.2-1B",            "Llama-3.2-1B"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="smollm2", choices=["smollm2", "qwen3", "llama"])
    p.add_argument("--size", default="135M")
    p.add_argument("--base", action="store_true", help="(SmolLM2 only) fetch base instead of instruct")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variant = "base" if args.base else "instruct"
    key = (args.model, args.size, variant)
    if key not in MODELS:
        raise SystemExit(
            f"unknown model: {key}. Available:\n"
            + "\n".join(f"  --model {k[0]} --size {k[1]}" + (" --base" if k[2] == "base" else "")
                        for k in MODELS)
        )

    repo_id, dirname = MODELS[key]
    local_dir = REPO_ROOT / "models" / dirname

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer.model",
            "*.txt",
            "*.md",
            "*.tiktoken",       # Qwen3 uses tiktoken-format vocab
            "merges.txt",
        ],
    )

    config_path = local_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        print("\n=== config.json ===")
        for k in sorted(config):
            print(f"  {k}: {config[k]}")

    print("\n=== safetensors tensors ===")
    st_files = sorted(local_dir.glob("*.safetensors"))
    st_files = [p for p in st_files if "fp32" not in p.parts]
    total_params = 0
    total_bytes = 0
    for st_file in st_files:
        with safe_open(st_file, framework="pt") as f:
            print(f"\n[{st_file.name}] keys={len(f.keys())}")
            for key in f.keys():
                t = f.get_tensor(key)
                numel = t.numel()
                total_params += numel
                total_bytes += t.element_size() * numel
                print(f"  {key:60s} {str(list(t.shape)):20s} {t.dtype}")

    print(f"\n=== totals ===")
    print(f"  parameters: {total_params:,}")
    print(f"  bytes     : {total_bytes:,} ({total_bytes / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
