#!/usr/bin/env python3
"""Audit CMExam split overlap and produce hashes without exposing examples."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CMEXAM = REPO_ROOT / "phase1/data/processed/cmexam"
OUTPUT = REPO_ROOT / "phase1/audit/benchmark_split_audit.json"


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


def fmt_prompt(row: dict[str, Any]) -> str:
    options = "\n".join(f"{item['key']}. {item['value']}" for item in row["Options"])
    return f"{row['Question']}\n{options}\n答案："


def aggregate_csv_hash(root: Path) -> tuple[int, str]:
    files = sorted(root.glob("*.csv"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def prompt_question_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        normalize(str(row.get("prompt", "")).split("\n", 1)[0])
        for row in read_jsonl(path)
        if row.get("prompt")
    }


def main() -> None:
    split_rows = {name: read_jsonl(CMEXAM / f"{name}.jsonl") for name in ("train", "valid", "test")}
    question_sets = {
        name: {normalize(str(row.get("Question", ""))) for row in rows if row.get("Question")}
        for name, rows in split_rows.items()
    }
    within_duplicates = {
        name: sum(count - 1 for count in Counter(normalize(str(row.get("Question", ""))) for row in rows).values() if count > 1)
        for name, rows in split_rows.items()
    }

    current_holdout = read_jsonl(CMEXAM / "holdout.jsonl")
    holdout_prompts = {normalize(str(row["prompt"])) for row in current_holdout}
    holdout_questions = {normalize(str(row["prompt"]).split("\n", 1)[0]) for row in current_holdout}
    test_prompts = {normalize(fmt_prompt(row)) for row in split_rows["test"] if len(str(row.get("Answer", "")).strip()) == 1}
    grpo_train = read_jsonl(CMEXAM / "grpo_train.jsonl")
    grpo_train_questions = {normalize(str(row["prompt"]).split("\n", 1)[0]) for row in grpo_train}
    downstream_training_paths = [
        REPO_ROOT / "phase1/results/week19_distill/data/real_sft.jsonl",
        REPO_ROOT / "phase1/results/week19_distill/data/distill_sft.jsonl",
        REPO_ROOT / "phase1/results/week19_distill/data/mixed_sft.jsonl",
        REPO_ROOT / "phase1/results/week20_distill/data/rs_mcq_sft.jsonl",
        REPO_ROOT / "phase1/results/week20_distill/data/rs_teacher_sft.jsonl",
        REPO_ROOT / "phase1/results/week20_distill/data/rs_both_sft.jsonl",
        REPO_ROOT / "phase1/results/week21_synthetic/data/replacement_50.jsonl",
    ]
    downstream_overlap = {
        str(path.relative_to(REPO_ROOT)): {
            "records": len(read_jsonl(path)),
            "normalized_question_overlap_with_historical_holdout": len(
                prompt_question_set(path) & holdout_questions
            ),
            "normalized_question_overlap_with_official_test": len(
                prompt_question_set(path) & question_sets["test"]
            ),
        }
        for path in downstream_training_paths
        if path.exists()
    }

    cmmlu_test_count, cmmlu_test_hash = aggregate_csv_hash(REPO_ROOT / "phase1/data/cmmlu_local/test")
    cmmlu_dev_count, cmmlu_dev_hash = aggregate_csv_hash(REPO_ROOT / "phase1/data/cmmlu_local/dev")

    report = {
        "schema_version": "1.0",
        "created_at": "2026-08-28",
        "normalization": "NFKC + casefold + collapsed whitespace",
        "cmexam": {
            "splits": {
                name: {
                    "path": f"phase1/data/processed/cmexam/{name}.jsonl",
                    "records": len(rows),
                    "sha256": sha256_file(CMEXAM / f"{name}.jsonl"),
                    "within_split_duplicate_questions": within_duplicates[name],
                }
                for name, rows in split_rows.items()
            },
            "question_overlap": {
                "train_valid": len(question_sets["train"] & question_sets["valid"]),
                "train_test": len(question_sets["train"] & question_sets["test"]),
                "valid_test": len(question_sets["valid"] & question_sets["test"]),
            },
            "historical_holdout": {
                "path": "phase1/data/processed/cmexam/holdout.jsonl",
                "records": len(current_holdout),
                "sha256": sha256_file(CMEXAM / "holdout.jsonl"),
                "matched_to_official_test_prompts": len(holdout_prompts & test_prompts),
                "normalized_question_overlap_with_full_train": len(holdout_questions & question_sets["train"]),
                "normalized_question_overlap_with_grpo_train": len(holdout_questions & grpo_train_questions),
                "status": "DEVELOPMENT_CONTAMINATED",
            },
            "downstream_training_overlap": downstream_overlap,
        },
        "cmmlu": {
            "test": {"csv_files": cmmlu_test_count, "aggregate_sha256": cmmlu_test_hash},
            "dev": {"csv_files": cmmlu_dev_count, "aggregate_sha256": cmmlu_dev_hash},
            "status": "DEVELOPMENT_PUBLIC_REUSED",
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
