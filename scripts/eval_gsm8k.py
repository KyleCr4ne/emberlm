"""GSM8K evaluation harness for ctd_generate.

Usage:
    uv run scripts/eval_gsm8k.py --weights models/SmolLM2-360M-Instruct/fp32/model.safetensors --mode base --n 200
    uv run scripts/eval_gsm8k.py --weights runs/opus_sft_360m_v1/final.safetensors --mode sft --n 200 --thinking-budget 1024

Outputs accuracy, sample correct/wrong completions, and a JSONL log under runs_eval/.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from _ctd_common import detect_family, ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CTD_GEN = REPO_ROOT / "build" / "bin" / "ctd_generate"
OUT_DIR = REPO_ROOT / "runs_eval"


def _find_qwen3_tok_dir():
    for name in ("Qwen3-0.6B", "Qwen3-1.7B"):
        d = REPO_ROOT / "models" / name
        if d.exists():
            return d
    return REPO_ROOT / "models" / "Qwen3-0.6B"

_TOK_DIRS = {
    "smollm2": REPO_ROOT / "models" / "SmolLM2-135M-Instruct",
    "qwen3":   _find_qwen3_tok_dir(),
}

# Ask the model to terminate with `#### N` — the same delimiter GSM8K gold
# solutions use — so gold and prediction share the same extraction regex.
ANSWER_FORMAT = (
    "End your response with a line of exactly the form '#### N' where N is the "
    "integer answer (no units, no commas, nothing after)."
)
SYS_BASE = (
    "You are a math tutor. Solve the problem and give the answer.\n" + ANSWER_FORMAT
)
SYS_SFT = (
    "You are a careful math tutor. Think step by step inside <think>...</think>, "
    "then state the final answer.\n" + ANSWER_FORMAT
)

ANSWER_RE = re.compile(r"####\s*(-?\d[\d,]*)")
# Fallbacks when the model ignores the format directive.
ANSWER_PHRASE_RE = re.compile(r"answer\s*(?:is|:)\s*\$?\s*(-?\d[\d,]*)", re.IGNORECASE)
LAST_NUM_RE = re.compile(r"(-?\d[\d,]*)")


def _to_int(s: str) -> int | None:
    try: return int(s.replace(",", ""))
    except ValueError: return None


def gold_answer(solution: str) -> int | None:
    m = ANSWER_RE.search(solution)
    return _to_int(m.group(1)) if m else None


def predicted_answer(text: str) -> int | None:
    # Strip <think> block — intermediate numbers in the scratchpad aren't the answer.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Use findall + [-1]: the model may emit `#### N` more than once; last wins.
    matches = ANSWER_RE.findall(text)
    if matches:
        return _to_int(matches[-1])
    m = ANSWER_PHRASE_RE.search(text)
    if m:
        v = _to_int(m.group(1))
        if v is not None: return v
    nums = LAST_NUM_RE.findall(text)
    return _to_int(nums[-1]) if nums else None


def run_one(mcfg: ModelConfig, weights: Path, question: str, sys_prompt: str,
            max_new_tokens: int, temperature: float, top_k: int, top_p: int,
            seed: int, thinking_budget: int) -> tuple[str, str]:
    """Synchronous one-shot. Returns (raw_completion_with_specials, decoded_skip_special)."""
    tok = mcfg.tokenizer
    text = tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"): ids = ids.tolist()

    payload = {
        "input_ids": ids,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "eos_ids": mcfg.eos_ids,
        "max_seq_len": 4096,
        "thinking_token_budget": thinking_budget,
        "think_open_ids": mcfg.think_open_ids,
        "think_close_ids": mcfg.think_close_ids,
    }
    proc = subprocess.Popen(
        [str(CTD_GEN), "--model", str(weights)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    proc.stdin.write(json.dumps(payload).encode() + b"\n")
    proc.stdin.close()
    generated: list[int] = []
    while True:
        line = proc.stdout.readline()
        if not line: break
        try: evt = json.loads(line)
        except json.JSONDecodeError: continue
        if "token" in evt:
            generated.append(int(evt["token"]))
        elif evt.get("done"):
            break
    proc.wait()
    raw = tok.decode(generated, skip_special_tokens=False)
    clean = tok.decode(generated, skip_special_tokens=True)
    return raw, clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--mode", choices=["base", "sft"], required=True)
    ap.add_argument("--n", type=int, default=200, help="subsample size; 0 = full test set")
    ap.add_argument("--seed", type=int, default=42, help="RNG for subsampling AND generation")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default 512 for base, 2048 for sft")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="default 0 for base, 1024 for sft")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default 0.0 for base (greedy), 0.6 for sft")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    if args.mode == "base":
        max_new = args.max_new_tokens or 512
        think_budget = 0
        temp = 0.0 if args.temperature is None else args.temperature
        sys_prompt = SYS_BASE
    else:
        max_new = args.max_new_tokens or 2048
        think_budget = 1024 if args.thinking_budget is None else args.thinking_budget
        temp = 0.6 if args.temperature is None else args.temperature
        sys_prompt = SYS_SFT

    print(f"[eval_gsm8k] mode={args.mode} weights={args.weights}")
    print(f"[eval_gsm8k] T={temp} max_new={max_new} think_budget={think_budget} n={args.n}")

    family = detect_family(args.weights)
    tok_dir = _TOK_DIRS[family]
    mcfg = ModelConfig(family)
    tok = mcfg.tokenizer
    print(f"[eval_gsm8k] family={family}")
    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    if args.n > 0 and args.n < len(ds):
        rng.shuffle(indices)
        indices = indices[:args.n]
    print(f"[eval_gsm8k] {len(indices)} questions of {len(ds)} total")

    OUT_DIR.mkdir(exist_ok=True)
    out_name = f"gsm8k_{args.mode}_{args.weights.parent.name}_{args.weights.stem}.jsonl"
    out_path = OUT_DIR / out_name
    log = out_path.open("w")

    correct = 0
    n_ok_pred = 0
    rows: list[dict] = []
    t0 = time.time()
    for i, idx in enumerate(tqdm(indices, dynamic_ncols=True)):
        ex = ds[idx]
        gold = gold_answer(ex["answer"])
        if gold is None:
            continue
        try:
            raw, clean = run_one(
                mcfg, args.weights, ex["question"], sys_prompt,
                max_new, temp, args.top_k, args.top_p,
                seed=args.seed + i, thinking_budget=think_budget,
            )
        except Exception as e:
            print(f"\n[error idx={idx}]: {e}", file=sys.stderr)
            continue
        pred = predicted_answer(clean)
        is_ok = (pred is not None and pred == gold)
        n_ok_pred += int(pred is not None)
        correct += int(is_ok)
        row = {
            "idx": idx,
            "question": ex["question"],
            "gold": gold,
            "pred": pred,
            "correct": is_ok,
            "completion": clean,
            "raw_with_specials": raw,
        }
        rows.append(row)
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()

    log.close()
    elapsed = time.time() - t0
    n = len(rows)
    acc = correct / n if n else 0.0
    parse_rate = n_ok_pred / n if n else 0.0
    print()
    print(f"[eval_gsm8k] accuracy = {correct}/{n} = {acc:.3f}  "
          f"parse_rate={parse_rate:.3f}  elapsed={elapsed:.1f}s  "
          f"({elapsed/max(n,1):.2f}s/Q)")
    print(f"[eval_gsm8k] log → {out_path}")

    correct_rows = [r for r in rows if r["correct"]]
    wrong_rows = [r for r in rows if not r["correct"]]
    rng.shuffle(correct_rows)
    rng.shuffle(wrong_rows)

    def dump(label: str, rs: list[dict], k: int = 3):
        print(f"\n========== {label} (showing {min(k, len(rs))}) ==========")
        for r in rs[:k]:
            print(f"\n--- idx={r['idx']}  gold={r['gold']}  pred={r['pred']} ---")
            print(f"Q: {r['question']}")
            print(f"A: {r['completion'][:600]}{'…' if len(r['completion'])>600 else ''}")

    dump("CORRECT", correct_rows, k=3)
    dump("WRONG",   wrong_rows,   k=3)


if __name__ == "__main__":
    main()
