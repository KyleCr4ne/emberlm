"""Run a model in HF Transformers and capture intermediate activations.

Captures are saved as a safetensors file keyed by activation name, plus a JSON
with the prompt/input_ids/config so the C++ side can reproduce the same inputs.
Used to diff our C++ values tensor-by-tensor against a trusted reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else "SmolLM2-135M-Instruct"
MODEL_DIR = REPO_ROOT / "models" / MODEL_DIR_NAME
OUT_DIR = MODEL_DIR / "reference"

PROMPT = "The capital of France is"
SEED = 0

# fp32 on CPU: deterministic, no fused-kernel fast paths. Weights load as bf16
# and are upcast at load time.
DTYPE = torch.float32
DEVICE = "cpu"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    print(f"[ref] loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DTYPE)
    model.to(DEVICE)
    model.eval()

    captures: dict[str, torch.Tensor] = {}

    def save_output(name: str):
        def hook(_mod, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            captures[name] = tensor.detach().clone().contiguous()
        return hook

    hooks = []
    hooks.append(model.model.embed_tokens.register_forward_hook(save_output("embed_tokens")))
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(save_output(f"layer_{i:02d}_out")))
    hooks.append(model.model.norm.register_forward_hook(save_output("final_norm")))

    # Fine-grained captures for layer 0 to localize block-level failures.
    layer0 = model.model.layers[0]
    hooks.append(layer0.input_layernorm.register_forward_hook(save_output("layer_00_post_ln1")))
    hooks.append(layer0.self_attn.register_forward_hook(save_output("layer_00_post_attn")))
    hooks.append(
        layer0.post_attention_layernorm.register_forward_hook(save_output("layer_00_post_ln2"))
    )
    hooks.append(layer0.mlp.register_forward_hook(save_output("layer_00_post_mlp")))

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    print(f"[ref] prompt: {PROMPT!r}")
    print(f"[ref] input_ids: {input_ids.tolist()[0]}")
    print(f"[ref] tokens: {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    logits = out.logits.detach().contiguous()
    captures["input_ids"] = input_ids.to(torch.int64)
    captures["logits"] = logits

    print("\n=== captured activations ===")
    for name, t in captures.items():
        print(f"  {name:25s} {str(list(t.shape)):20s} {t.dtype}")

    top = logits[0, -1].topk(5)
    print("\n=== top-5 next-token predictions ===")
    for p, tok_id in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {tok_id:6d} {tokenizer.decode([tok_id])!r:20s} logit={p:.4f}")

    tensors_path = OUT_DIR / "activations.safetensors"
    save_file(captures, str(tensors_path))
    print(f"\n[ref] wrote {tensors_path} ({tensors_path.stat().st_size / 1024**2:.2f} MiB)")

    meta = {
        "prompt": PROMPT,
        "input_ids": input_ids.tolist()[0],
        "seed": SEED,
        "dtype": str(DTYPE).replace("torch.", ""),
        "device": DEVICE,
        "model_dir": str(MODEL_DIR.relative_to(REPO_ROOT)),
        "captured_keys": list(captures.keys()),
    }
    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[ref] wrote {meta_path}")


if __name__ == "__main__":
    main()
