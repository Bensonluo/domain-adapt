"""Create a deterministic, holdout-clean set of 100 CMExam seed instructions."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import normalize_mcq, read_jsonl, stable_id, write_jsonl

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=REPO_ROOT / "phase1/data/processed/cmexam/train.jsonl")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "phase1/data/processed/cmexam/test.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "phase1/data/raw/seed_instructions.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    holdout_questions = {
        str(row.get("Question", "")).strip() for row in read_jsonl(args.holdout)
        if str(row.get("Question", "")).strip()
    }
    pool = []
    for raw in read_jsonl(args.train):
        normalized, error = normalize_mcq(raw)
        if error or normalized["question"] in holdout_questions:
            continue
        normalized["seed_id"] = f"seed-{stable_id(normalized['question'])}"
        normalized["source"] = "CMExam/train"
        pool.append(normalized)
    random.Random(args.seed).shuffle(pool)
    if len(pool) < args.n:
        raise SystemExit(f"need {args.n} valid seeds, found {len(pool)}")
    selected = pool[: args.n]
    write_jsonl(args.output, selected)
    print(f"[seeds] wrote {len(selected)} holdout-clean seeds -> {args.output}")


if __name__ == "__main__":
    main()
