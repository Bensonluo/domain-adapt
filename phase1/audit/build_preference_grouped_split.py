#!/usr/bin/env python3
"""Build a deterministic prompt-grouped preference train/dev split.

Historical Week 15 files are intentionally left untouched. This script writes a
new version so old experiments remain reproducible while confirmatory runs can
use a split with zero normalized-prompt overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "phase1/data/processed/preference/train.jsonl"
DEFAULT_TRAIN = REPO_ROOT / "phase1/data/processed/preference/train_grouped_v1.jsonl"
DEFAULT_DEV = REPO_ROOT / "phase1/data/processed/preference/dev_grouped_v1.jsonl"
DEFAULT_AUDIT = REPO_ROOT / "phase1/audit/preference_split_audit.json"
HISTORICAL_TRAIN = REPO_ROOT / "phase1/data/processed/preference/train_split.jsonl"
HISTORICAL_HOLDOUT = REPO_ROOT / "phase1/data/processed/preference/holdout.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def source_id(row: dict[str, Any], default_source: str) -> str:
    raw = row.get("source") or row.get("source_id") or default_source
    return normalize_text(str(raw))


def prompt_group_key(row: dict[str, Any], default_source: str) -> str:
    payload = source_id(row, default_source) + "\0" + normalize_text(str(row["prompt"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_row_key(row: dict[str, Any], default_source: str) -> str:
    fields = [
        source_id(row, default_source),
        normalize_text(str(row.get("prompt", ""))),
        normalize_text(str(row.get("chosen", ""))),
        normalize_text(str(row.get("rejected", ""))),
    ]
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"prompt", "chosen", "rejected"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def split_by_group(
    rows: list[dict[str, Any]], default_source: str, target_dev_rows: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[prompt_group_key(row, default_source)].append(index)

    group_keys = sorted(groups)
    random.Random(seed).shuffle(group_keys)
    dev_groups: set[str] = set()
    dev_count = 0
    for key in group_keys:
        if dev_count >= target_dev_rows:
            break
        dev_groups.add(key)
        dev_count += len(groups[key])

    train = [row for row in rows if prompt_group_key(row, default_source) not in dev_groups]
    dev = [row for row in rows if prompt_group_key(row, default_source) in dev_groups]
    return train, dev, groups


def overlap_report(
    train: list[dict[str, Any]], dev: list[dict[str, Any]], default_source: str
) -> dict[str, int]:
    train_groups = {prompt_group_key(row, default_source) for row in train}
    dev_groups = {prompt_group_key(row, default_source) for row in dev}
    train_rows = {canonical_row_key(row, default_source) for row in train}
    dev_rows = {canonical_row_key(row, default_source) for row in dev}
    return {
        "normalized_prompt_group_overlap": len(train_groups & dev_groups),
        "canonical_row_overlap": len(train_rows & dev_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--target-dev-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--source-id", default="modelzhang/medical_evidence_DPO")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    train, dev, groups = split_by_group(rows, args.source_id, args.target_dev_rows, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.dev_output, dev)

    group_sizes = sorted((len(indices) for indices in groups.values()), reverse=True)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": "2026-08-28",
        "policy": "NFKC + casefold + collapsed whitespace; group by source_id and normalized prompt",
        "seed": args.seed,
        "target_dev_rows": args.target_dev_rows,
        "source_id_default": args.source_id,
        "input": {
            "path": str(args.input.relative_to(REPO_ROOT)),
            "sha256": sha256_file(args.input),
            "records": len(rows),
            "prompt_groups": len(groups),
            "duplicate_prompt_records": len(rows) - len(groups),
            "max_prompt_group_size": group_sizes[0] if group_sizes else 0,
        },
        "outputs": {
            "train": {
                "path": str(args.train_output.relative_to(REPO_ROOT)),
                "records": len(train),
                "sha256": sha256_file(args.train_output),
            },
            "dev": {
                "path": str(args.dev_output.relative_to(REPO_ROOT)),
                "records": len(dev),
                "sha256": sha256_file(args.dev_output),
            },
        },
        "new_split_overlap": overlap_report(train, dev, args.source_id),
        "historical_random_split": None,
    }

    if HISTORICAL_TRAIN.exists() and HISTORICAL_HOLDOUT.exists():
        historical_train = read_jsonl(HISTORICAL_TRAIN)
        historical_holdout = read_jsonl(HISTORICAL_HOLDOUT)
        report["historical_random_split"] = {
            "train_path": str(HISTORICAL_TRAIN.relative_to(REPO_ROOT)),
            "dev_path": str(HISTORICAL_HOLDOUT.relative_to(REPO_ROOT)),
            "train_records": len(historical_train),
            "dev_records": len(historical_holdout),
            **overlap_report(historical_train, historical_holdout, args.source_id),
        }

    if report["new_split_overlap"]["normalized_prompt_group_overlap"] != 0:
        raise RuntimeError("grouped split invariant failed: prompt overlap is non-zero")
    if report["new_split_overlap"]["canonical_row_overlap"] != 0:
        raise RuntimeError("grouped split invariant failed: row overlap is non-zero")

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
