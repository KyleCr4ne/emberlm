"""One-shot CLI chat against a ctd_generate checkpoint.

Prints raw decoded text including any <think>...</think> block the model emits.

Usage:
    uv run scripts/quick_chat.py --weights runs/opus_sft_sanity/final.safetensors \
        --prompt "What is 23 * 47?" --max-new-tokens 256
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _ctd_common import detect_family, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CTD_GEN = REPO_ROOT / "build" / "bin" / "ctd_generate"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="Force </think> after N tokens inside <think>; 0 disables.")
    args = ap.parse_args()

    mcfg = ModelConfig(detect_family(Path(args.weights)))
    tok = mcfg.tokenizer
    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    payload = {
        "input_ids": ids,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "eos_ids": mcfg.eos_ids,
        "max_seq_len": 4096,
        "thinking_token_budget": args.thinking_budget,
        "think_open_ids": mcfg.think_open_ids,
        "think_close_ids": mcfg.think_close_ids,
    }

    proc = subprocess.Popen(
        [str(CTD_GEN), "--model", args.weights],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=False,
    )
    proc.stdin.write(json.dumps(payload).encode() + b"\n")
    proc.stdin.close()

    generated: list[int] = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "token" in evt:
            generated.append(int(evt["token"]))
        elif evt.get("done"):
            print(f"\n[done: {evt.get('reason')}]", file=sys.stderr)
            break

    completion = tok.decode(generated, skip_special_tokens=False)
    print(completion)


if __name__ == "__main__":
    main()
