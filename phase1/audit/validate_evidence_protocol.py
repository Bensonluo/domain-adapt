#!/usr/bin/env python3
"""Fail-fast validator for Phase 1 evaluation isolation invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "phase1/audit"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def check_path_hash(relative_path: str, expected: str) -> None:
    path = REPO_ROOT / relative_path
    require(path.is_file(), f"missing artifact: {relative_path}")
    actual = sha256_file(path)
    require(actual == expected, f"hash drift: {relative_path}: {actual} != {expected}")


def main() -> None:
    preference = load(AUDIT / "preference_split_audit.json")
    benchmark = load(AUDIT / "benchmark_split_audit.json")
    candidate = load(AUDIT / "cmexam_confirmation_candidate.json")
    replacement = load(AUDIT / "week21_clean_replacement_audit.json")
    week17 = load(AUDIT / "week17_clean_reanalysis.json")

    require(preference["new_split_overlap"]["normalized_prompt_group_overlap"] == 0,
            "new preference split has prompt-group overlap")
    require(preference["new_split_overlap"]["canonical_row_overlap"] == 0,
            "new preference split has row overlap")
    require(preference["historical_random_split"]["normalized_prompt_group_overlap"] == 50,
            "historical preference leakage count drifted; regenerate audit and review")
    check_path_hash(preference["outputs"]["train"]["path"], preference["outputs"]["train"]["sha256"])
    check_path_hash(preference["outputs"]["dev"]["path"], preference["outputs"]["dev"]["sha256"])

    holdout = benchmark["cmexam"]["historical_holdout"]
    require(holdout["status"] == "DEVELOPMENT_CONTAMINATED", "historical CMExam status changed")
    require(holdout["normalized_question_overlap_with_grpo_train"] == 8,
            "GRPO direct overlap count drifted; regenerate audit and review")
    require(holdout["normalized_question_overlap_with_full_train"] == 26,
            "CMExam train overlap count drifted; regenerate audit and review")

    require(candidate["status"] == "FROZEN_CONFIRMATION_CANDIDATE_NOT_BLIND",
            "confirmation candidate must not be labeled blind")
    require(all(value == 0 for value in candidate["invariants"].values()),
            "confirmation candidate contamination invariant failed")
    check_path_hash(candidate["output"]["path"], candidate["output"]["sha256"])

    replacement_invariants = replacement["invariants"]
    require(replacement_invariants["equal_arm_records"], "Week 21 clean arms differ in size")
    require(replacement_invariants["equal_slot_ids"], "Week 21 clean arm slots differ")
    require(replacement_invariants["equal_label_distribution"], "Week 21 clean label distributions differ")
    require(replacement_invariants["control_prohibited_overlap"] == 0,
            "Week 21 clean control overlaps confirmation candidate")
    require(replacement_invariants["treatment_prohibited_overlap"] == 0,
            "Week 21 clean treatment overlaps confirmation candidate")
    require(replacement_invariants["completion_budget_relative_difference_le_1pct"],
            "Week 21 clean completion budgets differ by more than 1%")
    require(replacement["inputs"]["confirmation"]["sha256"] == candidate["output"]["sha256"],
            "Week 21 clean builder used a different confirmation candidate")
    check_path_hash(replacement["outputs"]["control"]["path"],
                    replacement["outputs"]["control"]["sha256"])
    check_path_hash(replacement["outputs"]["treatment"]["path"],
                    replacement["outputs"]["treatment"]["sha256"])

    require(week17["training_overlap"]["overlap_count"] == 8,
            "Week 17 GRPO overlap count drifted")
    require(week17["retained_prefix_reanalysis"]["n"] == 49,
            "Week 17 recoverable clean prefix size drifted")
    require(week17["full_500_clean_effect_partial_identification"]["status"] == "BOUNDED_NOT_POINT_IDENTIFIED",
            "Week 17 full clean effect must remain bounded, not point-identified")
    require(week17["evidence_gap"]["missing_paired_prediction_rows_per_arm"] == 450,
            "Week 17 retained prediction gap drifted")
    require(week17["claim_disposition"]["historical_true_transfer_claim"] == "INVALIDATED",
            "Week 17 historical claim was upgraded without confirmation")
    for artifact in week17["inputs"].values():
        check_path_hash(artifact["path"], artifact["sha256"])

    print("PASS: Phase 1 evidence protocol invariants hold")


if __name__ == "__main__":
    main()
