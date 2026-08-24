"""Merge a PEFT adapter into its base checkpoint for fixed holdout evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MTL_TIMEOUT", "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.output)
    print(f"[merge] {args.adapter} + {args.base} -> {args.output}")


if __name__ == "__main__":
    main()
