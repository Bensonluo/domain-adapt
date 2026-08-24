"""Paired uncertainty analysis for the 50% synthetic replacement experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_mcnemar_p(control_only: int, treatment_only: int) -> float:
    discordant = control_only + treatment_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(control_only, treatment_only) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=-0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    control = read_jsonl(args.control)
    treatment = read_jsonl(args.treatment)
    if len(control) != len(treatment) or not control:
        raise ValueError("paired prediction files must have the same non-zero length")
    for index, (left, right) in enumerate(zip(control, treatment)):
        if left.get("index") != index or right.get("index") != index or left.get("gold") != right.get("gold"):
            raise ValueError(f"prediction pairing mismatch at index {index}")
        if not isinstance(left.get("correct"), bool) or not isinstance(right.get("correct"), bool):
            raise ValueError(f"non-boolean correctness at index {index}")

    differences = [int(right["correct"]) - int(left["correct"]) for left, right in zip(control, treatment)]
    delta = sum(differences) / len(differences)
    rng = random.Random(args.seed)
    bootstraps = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(args.bootstrap_samples)
    )
    lower = bootstraps[int(0.025 * args.bootstrap_samples)]
    upper = bootstraps[int(0.975 * args.bootstrap_samples) - 1]
    control_only = sum(bool(left["correct"]) and not bool(right["correct"]) for left, right in zip(control, treatment))
    treatment_only = sum(not bool(left["correct"]) and bool(right["correct"]) for left, right in zip(control, treatment))
    result = {
        "n_pairs": len(differences),
        "control_accuracy": sum(bool(row["correct"]) for row in control) / len(control),
        "treatment_accuracy": sum(bool(row["correct"]) for row in treatment) / len(treatment),
        "delta": delta,
        "noninferiority_margin": args.margin,
        "point_estimate_meets_margin": delta >= args.margin,
        "paired_bootstrap": {
            "samples": args.bootstrap_samples, "seed": args.seed,
            "confidence_level": 0.95, "delta_interval": [lower, upper],
            "noninferiority_established": lower > args.margin,
        },
        "mcnemar_exact": {
            "control_only_correct": control_only, "treatment_only_correct": treatment_only,
            "discordant": control_only + treatment_only,
            "two_sided_p": exact_mcnemar_p(control_only, treatment_only),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
