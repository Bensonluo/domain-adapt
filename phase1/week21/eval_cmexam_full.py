"""Deterministic CMExam evaluation retaining every paired prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / "phase1").is_dir())
sys.path.insert(0, str(REPO_ROOT / "phase1/week17"))
from reward_functions import extract_answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "phase1/data/processed/cmexam/holdout.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(device)
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(
                generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
            )
            for offset, (row, text) in enumerate(zip(batch, texts)):
                gold = str(row["answer"]).strip().upper()
                pred = extract_answer(text)
                predictions.append({
                    "index": start + offset,
                    "gold": gold,
                    "pred": pred,
                    "correct": pred == gold,
                    "gen_head": text[:120],
                })
            if start == 0 or (start + len(batch)) % 80 == 0:
                correct = sum(bool(row["correct"]) for row in predictions)
                print(f"[eval] {len(predictions)}/{len(rows)} accuracy={correct / len(predictions):.3f}", flush=True)

    correct = sum(bool(row["correct"]) for row in predictions)
    unparseable = sum(row["pred"] is None for row in predictions)
    result = {
        "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "n": len(predictions), "accuracy": correct / len(predictions), "correct": correct,
        "unparseable": unparseable, "unparseable_rate": unparseable / len(predictions),
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
        "decoding": "greedy", "device": device, "dtype": str(dtype),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_path = Path(str(args.output) + ".preds.jsonl")
    prediction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    print(f"[eval] accuracy={result['accuracy']:.3f} ({correct}/{len(predictions)}) -> {args.output}")


if __name__ == "__main__":
    main()
