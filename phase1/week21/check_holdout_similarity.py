"""Measure high-overlap train/synthetic questions against the complete holdout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import ngrams, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.78)
    args = parser.parse_args()

    holdout = [ngrams(str(row.get("Question", row.get("question", "")))) for row in read_jsonl(args.holdout)]
    inverted: dict[str, set[int]] = defaultdict(set)
    for index, grams in enumerate(holdout):
        for gram in grams:
            inverted[gram].add(index)

    def maximum(text: str) -> float:
        own = ngrams(text.split("\n", 1)[0])
        candidates: set[int] = set()
        for gram in own:
            candidates.update(inverted[gram])
        return max((len(own & holdout[i]) / len(own | holdout[i]) for i in candidates), default=0.0)

    data = args.sweep / "data"
    sources = {
        "seeds": (data / "seed_instructions.jsonl", "question"),
        "self": (data / "self_instruct.jsonl", "question"),
        "evol": (data / "evol_instruct.jsonl", "question"),
        "replacement_all": (data / "replacement_50.jsonl", "prompt"),
    }
    result: dict[str, object] = {"metric": "character_3gram_jaccard", "threshold": args.threshold, "sources": {}}
    for name, (path, key) in sources.items():
        rows = read_jsonl(path)
        values = sorted(maximum(str(row[key])) for row in rows)
        result["sources"][name] = {
            "n": len(values), "above_or_equal_threshold": sum(value >= args.threshold for value in values),
            "maximum": max(values, default=0.0),
            "p95": values[int(0.95 * (len(values) - 1))] if values else None,
        }
    replacement = read_jsonl(data / "replacement_50.jsonl")
    result["replacement_high_overlap_by_source"] = {
        source: sum(maximum(str(row["prompt"])) >= args.threshold for row in replacement if row.get("source") == source)
        for source in sorted({str(row.get("source")) for row in replacement})
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
