#!/usr/bin/env python3
"""Freeze a structurally clean CMExam confirmation candidate from valid.

The output is not called a blind test: CMExam is public and labels are locally
accessible. It is a confirmation candidate that has not been used for model
selection in this project and excludes normalized-question overlap with both
training and the already-contaminated test split.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CMEXAM = REPO_ROOT / "phase1/data/processed/cmexam"
OUTPUT = CMEXAM / "confirmation_candidate_v1.jsonl"
REPORT = REPO_ROOT / "phase1/audit/cmexam_confirmation_candidate.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_prompt_questions(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    questions: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        local = {
            normalize(str(row.get("prompt", "")).split("\n", 1)[0])
            for row in rows
            if row.get("prompt")
        }
        questions.update(local)
        counts[str(path.relative_to(REPO_ROOT))] = len(local)
    return questions, counts


def valid_single_choice(row: dict[str, Any]) -> bool:
    answer = str(row.get("Answer", "")).strip().upper()
    return bool(row.get("Question")) and bool(row.get("Options")) and len(answer) == 1 and answer in "ABCDE"


def format_prompt(row: dict[str, Any]) -> str:
    options = "\n".join(f"{item['key']}. {item['value']}" for item in row["Options"])
    return f"{row['Question']}\n{options}\n答案："


def main() -> None:
    train = read_jsonl(CMEXAM / "train.jsonl")
    valid = read_jsonl(CMEXAM / "valid.jsonl")
    test = read_jsonl(CMEXAM / "test.jsonl")

    extra_training_paths = [
        REPO_ROOT / "phase1/data/processed/preference/train.jsonl",
        REPO_ROOT / "phase1/data/processed/cmexam/grpo_train.jsonl",
        REPO_ROOT / "phase1/results/week19_distill/data/real_sft.jsonl",
        REPO_ROOT / "phase1/results/week20_distill/data/rs_mcq_sft.jsonl",
        REPO_ROOT / "phase1/results/week20_distill/data/rs_teacher_sft.jsonl",
        REPO_ROOT / "phase1/results/week20_distill/data/rs_both_sft.jsonl",
        REPO_ROOT / "phase1/results/week21_synthetic/data/replacement_50.jsonl",
    ]
    extra_training_questions, extra_training_counts = training_prompt_questions(extra_training_paths)

    train_questions = {normalize(str(row.get("Question", ""))) for row in train if row.get("Question")}
    test_questions = {normalize(str(row.get("Question", ""))) for row in test if row.get("Question")}

    seen: set[str] = set()
    kept: list[dict[str, str]] = []
    excluded_invalid = 0
    excluded_train_overlap = 0
    excluded_test_overlap = 0
    excluded_extra_training_overlap = 0
    excluded_valid_duplicate = 0

    for row in valid:
        if not valid_single_choice(row):
            excluded_invalid += 1
            continue
        question = normalize(str(row["Question"]))
        if question in train_questions:
            excluded_train_overlap += 1
            continue
        if question in test_questions:
            excluded_test_overlap += 1
            continue
        if question in extra_training_questions:
            excluded_extra_training_overlap += 1
            continue
        if question in seen:
            excluded_valid_duplicate += 1
            continue
        seen.add(question)
        item_id = hashlib.sha256(question.encode("utf-8")).hexdigest()[:20]
        kept.append(
            {
                "id": item_id,
                "prompt": format_prompt(row),
                "answer": str(row["Answer"]).strip().upper(),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")

    output_questions = {normalize(row["prompt"].split("\n", 1)[0]) for row in kept}
    report = {
        "schema_version": "1.0",
        "created_at": "2026-08-28",
        "status": "FROZEN_CONFIRMATION_CANDIDATE_NOT_BLIND",
        "reason_not_blind": [
            "CMExam is a public benchmark",
            "answer labels are locally accessible",
            "no external evaluator or label custodian is configured",
        ],
        "selection_policy": "valid split only; single-choice; exclude normalized-question overlap with train and contaminated test; deduplicate valid; no model-output selection",
        "source": {
            "path": "phase1/data/processed/cmexam/valid.jsonl",
            "sha256": sha256_file(CMEXAM / "valid.jsonl"),
            "records": len(valid),
        },
        "output": {
            "path": "phase1/data/processed/cmexam/confirmation_candidate_v1.jsonl",
            "sha256": sha256_file(OUTPUT),
            "records": len(kept),
        },
        "excluded": {
            "invalid_or_not_single_choice": excluded_invalid,
            "train_question_overlap": excluded_train_overlap,
            "contaminated_test_question_overlap": excluded_test_overlap,
            "other_phase1_training_prompt_overlap": excluded_extra_training_overlap,
            "within_valid_duplicate": excluded_valid_duplicate,
        },
        "audited_additional_training_sources": extra_training_counts,
        "invariants": {
            "normalized_question_overlap_with_train": len(output_questions & train_questions),
            "normalized_question_overlap_with_test": len(output_questions & test_questions),
            "normalized_question_overlap_with_other_phase1_training": len(
                output_questions & extra_training_questions
            ),
            "duplicate_output_ids": len(kept) - len({row["id"] for row in kept}),
        },
        "use_policy": {
            "allowed": "run only after candidate model, hyperparameters, seeds and analysis plan are frozen",
            "forbidden": "use results to tune and rerun the same confirmation claim",
            "research_exit": "does not replace an externally held blind test",
        },
    }

    if any(report["invariants"].values()):
        raise RuntimeError(f"confirmation candidate invariant failed: {report['invariants']}")

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
