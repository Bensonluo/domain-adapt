"""Build the controlled 50% real / 50% synthetic SFT treatment dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import normalize_mcq, prompt_from_mcq, read_jsonl, stable_id, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-sft", required=True, type=Path)
    parser.add_argument("--synthetic", required=True, type=Path, nargs="+")
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-total", type=int, default=2000)
    parser.add_argument("--synthetic-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    n_synthetic = round(args.n_total * args.synthetic_fraction)
    n_real = args.n_total - n_synthetic
    if n_real <= 0 or n_synthetic <= 0:
        parser.error("both real and synthetic partitions must be non-empty")

    real_rows = read_jsonl(args.real_sft)
    if len(real_rows) < n_real:
        raise SystemExit(f"need {n_real} real rows, found {len(real_rows)}")
    holdout_questions = {str(row.get("Question", row.get("question", ""))).strip() for row in read_jsonl(args.holdout)}
    candidates = []
    seen_questions = set()
    seen_sources = set()
    rejected_holdout = 0
    for path in args.synthetic:
        for raw in read_jsonl(path):
            normalized, error = normalize_mcq(raw)
            if error:
                continue
            question = normalized["question"]
            source_id = raw.get("source_id")
            synthetic_id = raw.get("synthetic_id", stable_id(question))
            if question in holdout_questions:
                rejected_holdout += 1
                continue
            if question in seen_questions or synthetic_id in seen_sources or (source_id and source_id in seen_sources):
                continue
            seen_questions.add(question)
            if source_id:
                seen_sources.add(source_id)
            candidates.append({**normalized, "synthetic_id": synthetic_id, "method": raw.get("method")})
    if len(candidates) < n_synthetic:
        raise SystemExit(f"need {n_synthetic} unique leakage-clean synthetic rows, found {len(candidates)}")

    rng = random.Random(args.seed)
    real_indices = list(range(len(real_rows)))
    rng.shuffle(real_indices)
    rng.shuffle(candidates)
    output = []
    for index in real_indices[:n_real]:
        row = dict(real_rows[index])
        row.update({"source": "real", "replacement_id": f"real-{row.get('question_id', index)}"})
        output.append(row)
    for row in candidates[:n_synthetic]:
        output.append({
            "prompt": prompt_from_mcq(row),
            "completion": f"{row['answer']}\n{row['explanation']}",
            "question_id": row["synthetic_id"], "gold": row["answer"],
            "source": "synthetic", "method": row.get("method"),
            "replacement_id": row["synthetic_id"],
        })
    rng.shuffle(output)
    write_jsonl(args.output, output)
    manifest = {
        "n_total": len(output), "n_real": n_real, "n_synthetic": n_synthetic,
        "synthetic_fraction": n_synthetic / len(output), "seed": args.seed,
        "real_sft": str(args.real_sft), "synthetic_inputs": [str(path) for path in args.synthetic],
        "holdout": str(args.holdout), "rejected_exact_holdout_matches": rejected_holdout,
        "unique_replacement_ids": len({row["replacement_id"] for row in output}),
        "synthetic_method_distribution": dict(sorted(Counter(
            row.get("method") for row in output if row.get("source") == "synthetic"
        ).items())),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[replacement] {n_real} real + {n_synthetic} synthetic -> {args.output}")


if __name__ == "__main__":
    main()
