"""Prepare SFT dataset for training.

Outputs (little-endian):
    {out_dir}/train.bin           int32 stream, N_train * seq_len tokens
    {out_dir}/train_mask.bin      int8  stream, parallel 0/1 loss mask
    {out_dir}/val.bin / val_mask.bin
    {out_dir}/doc_offsets.bin     int64, prefix offsets in tokens
    {out_dir}/meta.json           tokenizer/seq_len/special-id metadata + filter stats

Loss mask: 1 on assistant turn (incl. <think>...</think>), 0 on system/user/pad.
Off-by-one is handled by the dataloader (input=tokens[:-1], targets=tokens[1:],
loss_mask=mask[1:]).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "Roman1111111/claude-opus-4.6-10000x"

_FAMILIES = {
    "smollm2": {
        "tokenizer_dir": REPO_ROOT / "models" / "SmolLM2-135M-Instruct",
        "default_out": "data/opus_reasoning",
    },
    "qwen3": {
        "tokenizer_dir": REPO_ROOT / "models" / "Qwen3-0.6B",
        "default_out": "data/opus_reasoning_qwen3",
    },
    "llama": {
        "tokenizer_dir": REPO_ROOT / "models" / "Llama-3.2-1B-Instruct",
        "default_out": "data/opus_reasoning_llama",
    },
}


def compose_assistant_with_think(messages: list[dict]) -> list[dict] | None:
    """Wrap the `reasoning` field as <think>...</think> prepended to `content`.

    The dataset stores chain-of-thought in a separate `reasoning` field on the
    assistant message. Returns None if any message has None content (some rows
    in the dataset are broken).
    """
    if not messages:
        return None
    for m in messages:
        if m.get("content") is None:
            return None
    last = messages[-1]
    if last["role"] != "assistant":
        return messages
    reasoning = last.get("reasoning")
    if reasoning:
        composed = f"<think>\n{reasoning}\n</think>\n\n{last['content']}"
        return messages[:-1] + [{"role": "assistant", "content": composed}]
    return messages


def encode_example(raw_messages: list[dict], tokenizer, seq_len: int):
    """Return (full_ids, mask, ok) where ok=False means filter this example out.

    BPE boundary: tokenize messages[:-1] with add_generation_prompt=True to get
    the exact prefix length (ending with <|im_start|>assistant\\n). Rendering to
    text first then encoding the full string ensures BPE sees the full context
    across the prompt/assistant boundary — no stitching artifacts.
    """
    messages = compose_assistant_with_think(raw_messages)
    if messages is None:
        return None, None, False
    if messages[-1]["role"] != "assistant":
        return None, None, False

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False
    )
    full_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    if not full_text.startswith(prompt_text):
        return None, None, False
    prompt_only_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Sanity: full must extend prompt exactly. BPE merge drift across the boundary
    # would silently corrupt the mask, so fail loudly.
    if full_ids[: len(prompt_only_ids)] != prompt_only_ids:
        return None, None, False

    if len(full_ids) > seq_len:
        return None, None, False
    if len(full_ids) <= len(prompt_only_ids):
        return None, None, False

    mask = np.zeros(seq_len, dtype=np.int8)
    mask[len(prompt_only_ids) : len(full_ids)] = 1

    pad_id = tokenizer.pad_token_id
    padded = np.full(seq_len, pad_id, dtype=np.int32)
    padded[: len(full_ids)] = np.asarray(full_ids, dtype=np.int32)

    return padded, mask, True


def find_token_subseq(haystack: list[int], needle: list[int]) -> int:
    """Return start index of `needle` in `haystack`, or -1."""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(_FAMILIES), default="smollm2",
                    help="Model family — selects the tokenizer and default output dir.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override output dir (default: per-family).")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N examples (debug).")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-first", type=int, default=0,
                    help="Print first N kept examples with mask overlay, then exit.")
    args = ap.parse_args()

    fam = _FAMILIES[args.model]
    if args.out_dir is None:
        args.out_dir = Path(fam["default_out"])
    MODEL_DIR = fam["tokenizer_dir"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer from {MODEL_DIR} (family={args.model})")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # LLaMA 3.2 and some other models don't define pad_token.
        # Use EOS — padding is masked out by loss_mask anyway.
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = pad_id
        print(f"[load] no pad_token, using eos_token_id={pad_id} as pad")

    print(f"[load] dataset {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[load] {len(ds)} examples")

    first = ds[0]
    if "messages" not in first:
        print(f"[err] expected 'messages' field, got keys: {list(first.keys())}",
              file=sys.stderr)
        sys.exit(1)

    kept_tokens: list[np.ndarray] = []
    kept_masks: list[np.ndarray] = []
    n_too_long = 0
    n_other = 0

    for ex in tqdm(ds, desc="encoding"):
        toks, mask, ok = encode_example(ex["messages"], tokenizer, args.seq_len)
        if ok:
            kept_tokens.append(toks)
            kept_masks.append(mask)
        else:
            try:
                composed = compose_assistant_with_think(ex["messages"])
                if composed is None:
                    full_len = -1
                else:
                    full_text = tokenizer.apply_chat_template(
                        composed, add_generation_prompt=False, tokenize=False
                    )
                    full_len = len(tokenizer.encode(full_text, add_special_tokens=False))
            except Exception:
                full_len = -1
            if full_len > args.seq_len:
                n_too_long += 1
            else:
                n_other += 1

    n_kept = len(kept_tokens)
    n_total = len(ds)
    print(f"[filter] kept {n_kept}/{n_total} = {100.0*n_kept/n_total:.1f}%")
    print(f"[filter]   too_long: {n_too_long}, other: {n_other}")

    if n_kept == 0:
        print("[err] no examples kept", file=sys.stderr)
        sys.exit(1)

    if args.dump_first > 0:
        for i in range(min(args.dump_first, n_kept)):
            print(f"\n=== example {i} ===")
            toks, mask = kept_tokens[i], kept_masks[i]
            boundary = int(np.argmax(mask)) if mask.any() else 0
            lo = max(0, boundary - 6)
            hi = min(args.seq_len, boundary + 12)
            print(f"  boundary at idx={boundary}; showing [{lo}:{hi}]:")
            for j in range(lo, hi):
                tok_id = int(toks[j])
                tok_str = tokenizer.decode([tok_id])
                marker = ">>>" if j == boundary else "   "
                print(f"  {marker} idx={j:4d} mask={int(mask[j])} id={tok_id:6d} {tok_str!r}")
        return

    rng = random.Random(args.seed)
    idx = list(range(n_kept))
    rng.shuffle(idx)
    n_val = max(1, int(round(n_kept * args.val_frac)))
    val_idx = set(idx[:n_val])

    train_toks = np.stack([kept_tokens[i] for i in range(n_kept) if i not in val_idx])
    train_mask = np.stack([kept_masks[i]  for i in range(n_kept) if i not in val_idx])
    val_toks   = np.stack([kept_tokens[i] for i in range(n_kept) if i in val_idx])
    val_mask   = np.stack([kept_masks[i]  for i in range(n_kept) if i in val_idx])

    print(f"[split] train={len(train_toks)} val={len(val_toks)}")

    assert train_toks.dtype == np.int32 and train_mask.dtype == np.int8
    train_toks.tofile(args.out_dir / "train.bin")
    train_mask.tofile(args.out_dir / "train_mask.bin")
    val_toks.tofile(args.out_dir / "val.bin")
    val_mask.tofile(args.out_dir / "val_mask.bin")

    # doc_offsets.bin: multiples of seq_len for no-packing layout; kept for forward-compat.
    n_train = len(train_toks)
    offsets = np.arange(n_train + 1, dtype=np.int64) * args.seq_len
    offsets.tofile(args.out_dir / "doc_offsets.bin")

    # Locate canonical <think>/</think> token sequences from actual encoded examples
    # rather than tokenizing the literal strings (avoids BPE non-additivity).
    think_open_str = "<think>"
    think_close_str = "</think>"
    think_open_ids = []
    think_close_ids = []
    for i in range(min(200, n_kept)):
        toks_i = kept_tokens[i].tolist()
        mask_i = kept_masks[i]
        start = int(np.argmax(mask_i)) if mask_i.any() else 0
        end = int(np.where(mask_i == 1)[0].max()) + 1 if mask_i.any() else 0
        decoded = tokenizer.decode(toks_i[start:end])
        if think_open_str in decoded and think_close_str in decoded:
            for L in (5, 10, 15):
                cand = tokenizer.encode(think_open_str, add_special_tokens=False)
                if cand and find_token_subseq(toks_i, cand) != -1:
                    think_open_ids = cand
                    break
            for L in (5, 10, 15):
                cand = tokenizer.encode(think_close_str, add_special_tokens=False)
                if cand and find_token_subseq(toks_i, cand) != -1:
                    think_close_ids = cand
                    break
            if think_open_ids and think_close_ids:
                break

    meta = {
        "tokenizer_name": str(MODEL_DIR.name),
        "tokenizer_dir": str(MODEL_DIR),
        "dataset": DATASET_NAME,
        "seq_len": args.seq_len,
        "pad_id": int(pad_id),
        "eos_id": int(tokenizer.eos_token_id),
        "bos_id": int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None,
        "im_start_id": int(x) if (x := tokenizer.convert_tokens_to_ids("<|im_start|>")) is not None else None,
        "im_end_id":   int(x) if (x := tokenizer.convert_tokens_to_ids("<|im_end|>")) is not None else None,
        "think_open_ids":  [int(x) for x in think_open_ids],
        "think_close_ids": [int(x) for x in think_close_ids],
        "vocab_size": int(tokenizer.vocab_size),
        "num_train_docs": int(len(train_toks)),
        "num_val_docs":   int(len(val_toks)),
        "filter_stats": {
            "total_input": int(n_total),
            "kept": int(n_kept),
            "dropped_too_long": int(n_too_long),
            "dropped_other":    int(n_other),
        },
        "endianness": "little",
        "token_dtype": "int32",
        "mask_dtype": "int8",
        "split_seed": args.seed,
        "val_frac": args.val_frac,
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[write] {args.out_dir}/")
    for p in sorted(args.out_dir.iterdir()):
        print(f"        {p.name:24s} {p.stat().st_size:>12d} bytes")


if __name__ == "__main__":
    main()
