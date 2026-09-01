#!/usr/bin/env python3
"""Reanalyse recoverable Week 17 evidence after removing GRPO-train overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(prompt: str) -> str:
    question = prompt.split("\n", 1)[0]
    question = unicodedata.normalize("NFKC", question).casefold().strip()
    return re.sub(r"\s+", " ", question)


def exact_mcnemar(control_only: int, treatment_only: int) -> float:
    discordant = control_only + treatment_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(control_only, treatment_only) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_statistics(control_correct: list[bool], treatment_correct: list[bool], samples: int, seed: int) -> dict[str, Any]:
    if not control_correct or len(control_correct) != len(treatment_correct):
        raise ValueError("paired correctness vectors must be non-empty and equal length")
    n = len(control_correct)
    differences = [int(treatment) - int(control) for control, treatment in zip(control_correct, treatment_correct)]
    rng = random.Random(seed)
    boot = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    control_only = sum(control and not treatment for control, treatment in zip(control_correct, treatment_correct))
    treatment_only = sum(treatment and not control for control, treatment in zip(control_correct, treatment_correct))
    return {
        "n": n,
        "control_correct": sum(control_correct),
        "treatment_correct": sum(treatment_correct),
        "control_accuracy": sum(control_correct) / n,
        "treatment_accuracy": sum(treatment_correct) / n,
        "delta": sum(differences) / n,
        "paired_bootstrap_samples": samples,
        "paired_bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "mcnemar_exact": {
            "control_only_correct": control_only,
            "treatment_only_correct": treatment_only,
            "p_value": exact_mcnemar(control_only, treatment_only),
        },
    }


def analyse(
    holdout: list[dict[str, Any]],
    training: list[dict[str, Any]],
    control_predictions: list[dict[str, Any]],
    treatment_predictions: list[dict[str, Any]],
    control_aggregate: dict[str, Any],
    treatment_aggregate: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(control_predictions) != len(treatment_predictions):
        raise ValueError("retained prediction samples differ in length")
    retained_n = len(control_predictions)
    if retained_n > len(holdout):
        raise ValueError("more predictions than holdout rows")
    expected_golds = [str(row["answer"]).strip().upper() for row in holdout[:retained_n]]
    if [row.get("gold") for row in control_predictions] != expected_golds:
        raise ValueError("control predictions do not align to holdout prefix")
    if [row.get("gold") for row in treatment_predictions] != expected_golds:
        raise ValueError("treatment predictions do not align to holdout prefix")

    training_questions = {normalize_question(str(row["prompt"])) for row in training}
    overlap_indices = [
        index for index, row in enumerate(holdout)
        if normalize_question(str(row["prompt"])) in training_questions
    ]
    retained_overlap = [index for index in overlap_indices if index < retained_n]
    retained_overlap_set = set(retained_overlap)
    retained_clean = [index for index in range(retained_n) if index not in retained_overlap_set]
    control_correct = [control_predictions[index].get("pred") == expected_golds[index] for index in retained_clean]
    treatment_correct = [treatment_predictions[index].get("pred") == expected_golds[index] for index in retained_clean]

    full_n = int(control_aggregate["n"])
    if full_n != int(treatment_aggregate["n"]) or full_n != len(holdout):
        raise ValueError("aggregate counts do not match holdout")
    observed_correct_delta = int(treatment_aggregate["correct"]) - int(control_aggregate["correct"])
    overlap_n = len(overlap_indices)
    clean_n = full_n - overlap_n
    # Each removed paired row can contribute at most -1 or +1 to the correct-count delta.
    clean_delta_count_bounds = [observed_correct_delta - overlap_n, observed_correct_delta + overlap_n]

    return {
        "schema_version": "1.0",
        "created_at": "2026-08-28",
        "historical_aggregate": {
            "n": full_n,
            "control_correct": int(control_aggregate["correct"]),
            "treatment_correct": int(treatment_aggregate["correct"]),
            "delta": float(treatment_aggregate["accuracy"]) - float(control_aggregate["accuracy"]),
        },
        "training_overlap": {
            "normalization": "first prompt line; NFKC + casefold + collapsed whitespace",
            "overlap_count": overlap_n,
            "overlap_indices_zero_based": overlap_indices,
            "retained_sample_overlap_count": len(retained_overlap),
            "retained_sample_overlap_indices_zero_based": retained_overlap,
        },
        "full_500_clean_effect_partial_identification": {
            "status": "BOUNDED_NOT_POINT_IDENTIFIED",
            "clean_n": clean_n,
            "clean_treatment_minus_control_correct_count_bounds": clean_delta_count_bounds,
            "clean_accuracy_delta_bounds": [bound / clean_n for bound in clean_delta_count_bounds],
            "interpretation": "The positive point delta cannot be fully explained by the 8 overlapping rows if aggregate counts are trusted, but exact clean accuracy, CI and McNemar are unrecoverable without all paired predictions.",
        },
        "retained_prefix_reanalysis": {
            "status": "EXPLORATORY_SAMPLE_ONLY",
            "retained_predictions": retained_n,
            "excluded_overlap_rows": len(retained_overlap),
            **paired_statistics(control_correct, treatment_correct, bootstrap_samples, seed),
        },
        "evidence_gap": {
            "status": "NON_RECOVERABLE_FROM_RETAINED_ARTIFACTS",
            "missing_paired_prediction_rows_per_arm": full_n - retained_n,
            "cause": "Week 17 eval_cmexam.py intentionally persisted preds[:50] while aggregate metrics used all 500 rows.",
            "prohibited_inference": "Do not present the retained-prefix estimate or bounded full effect as a clean 500-question confirmation result.",
        },
        "claim_disposition": {
            "historical_true_transfer_claim": "INVALIDATED",
            "narrower_clean_point_direction": "PARTIALLY_IDENTIFIED_POSITIVE_IF_AGGREGATES_TRUSTED",
            "confirmation_status": "UNVERIFIED_SINGLE_SEED_DEVELOPMENT_DATA",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "phase1/data/processed/cmexam/holdout.jsonl")
    parser.add_argument("--training", type=Path, default=REPO_ROOT / "phase1/data/processed/cmexam/grpo_train.jsonl")
    parser.add_argument("--control-predictions", type=Path, default=REPO_ROOT / "phase1/results/week17_grpo/base_cmexam_holdout.json.preds.jsonl")
    parser.add_argument("--treatment-predictions", type=Path, default=REPO_ROOT / "phase1/results/week17_grpo/mcq_base_fused/cmexam_holdout.json.preds.jsonl")
    parser.add_argument("--control-aggregate", type=Path, default=REPO_ROOT / "phase1/results/week17_grpo/base_cmexam_holdout.json")
    parser.add_argument("--treatment-aggregate", type=Path, default=REPO_ROOT / "phase1/results/week17_grpo/mcq_base_fused/cmexam_holdout.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "phase1/audit/week17_clean_reanalysis.json")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    result = analyse(
        read_jsonl(args.holdout), read_jsonl(args.training),
        read_jsonl(args.control_predictions), read_jsonl(args.treatment_predictions),
        load_json(args.control_aggregate), load_json(args.treatment_aggregate),
        args.bootstrap_samples, args.seed,
    )
    result["inputs"] = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for name, path in {
            "holdout": args.holdout, "training": args.training,
            "control_predictions": args.control_predictions, "treatment_predictions": args.treatment_predictions,
            "control_aggregate": args.control_aggregate, "treatment_aggregate": args.treatment_aggregate,
        }.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
