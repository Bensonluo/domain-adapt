"""Build leakage-clean, label/length-matched control and synthetic50 data.

Historical Week 21 artifacts are never overwritten. The treatment keeps half
of the control rows unchanged and replaces the other half with synthetic rows
matched to the removed real row by answer label and nearest text length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import compact_text, max_similarity, ngrams, normalize_mcq, prompt_from_mcq, read_jsonl, stable_id, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_from_prompt(prompt: str) -> str:
    return prompt.split("\n", 1)[0].strip()


def sft_length(row: dict[str, Any]) -> int:
    return len(str(row["prompt"])) + len(str(row["completion"]))


def strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def to_real_sft(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    normalized, error = normalize_mcq(row)
    if error or normalized is None:
        return None
    return {
        "prompt": prompt_from_mcq(normalized),
        "completion": f"{normalized['answer']}\n{normalized['explanation']}",
        "gold": normalized["answer"],
        "source": "real",
        "source_id": f"cmexam-train-{source_index}",
    }


def to_synthetic_sft(row: dict[str, Any], fallback_index: int) -> dict[str, Any] | None:
    normalized, error = normalize_mcq(row)
    if error or normalized is None:
        return None
    source_id = str(row.get("synthetic_id") or row.get("source_id") or stable_id(normalized["question"], fallback_index))
    return {
        "prompt": prompt_from_mcq(normalized),
        "completion": f"{normalized['answer']}\n{normalized['explanation']}",
        "gold": normalized["answer"],
        "source": "synthetic",
        "method": row.get("method"),
        "source_id": source_id,
    }


def collect_clean_rows(
    candidates: list[dict[str, Any]],
    confirmation_grams: list[set[str]],
    threshold: float,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected = Counter()
    maximum = 0.0
    for row in candidates:
        question = question_from_prompt(str(row["prompt"]))
        key = compact_text(question)
        if not key or key in seen:
            rejected["duplicate_question"] += 1
            continue
        similarity = max_similarity(question, confirmation_grams)
        maximum = max(maximum, similarity)
        if similarity >= threshold:
            rejected["confirmation_similarity"] += 1
            continue
        seen.add(key)
        kept.append({**row, "confirmation_max_3gram_jaccard": similarity})
        if len(kept) >= limit:
            break
    return kept, dict(sorted(rejected.items())), maximum


def match_replacements(
    real_candidates: list[dict[str, Any]], synthetic: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real_candidates:
        by_label[str(row["gold"])].append(row)
    for rows in by_label.values():
        rows.sort(key=lambda row: (int(row["_match_length"]), str(row["source_id"])))

    pairs: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for synth in synthetic:
        label = str(synth["gold"])
        choices = by_label[label]
        if not choices:
            raise RuntimeError(f"not enough real rows with answer label {label}")
        target = int(synth["_match_length"])
        best_index = min(
            range(len(choices)),
            key=lambda index: (abs(int(choices[index]["_match_length"]) - target), str(choices[index]["source_id"])),
        )
        real = choices.pop(best_index)
        pairs.append((real, synth, abs(int(real["_match_length"]) - target)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-raw", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, nargs="+", required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--treatment-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--n-total", type=int, default=2000)
    parser.add_argument("--synthetic-fraction", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--real-pool-size", type=int, default=4000)
    parser.add_argument("--tokenizer", type=Path, help="Optional local tokenizer for completion-token matching")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    n_synthetic = round(args.n_total * args.synthetic_fraction)
    n_shared_real = args.n_total - n_synthetic
    if n_synthetic <= 0 or n_shared_real <= 0:
        parser.error("control/treatment must both contain real data and treatment synthetic data")
    if args.real_pool_size < n_shared_real + n_synthetic:
        parser.error("real-pool-size must cover shared and matched-replacement rows")

    confirmation_rows = read_jsonl(args.confirmation)
    confirmation_questions = [question_from_prompt(str(row["prompt"])) for row in confirmation_rows]
    confirmation_grams = [ngrams(question) for question in confirmation_questions]

    raw_real = read_jsonl(args.real_raw)
    real_candidates = [row for index, raw in enumerate(raw_real) if (row := to_real_sft(raw, index)) is not None]
    random.Random(args.seed).shuffle(real_candidates)
    clean_real, real_rejected, real_max = collect_clean_rows(
        real_candidates, confirmation_grams, args.similarity_threshold, args.real_pool_size
    )
    if len(clean_real) < args.real_pool_size:
        raise SystemExit(f"need {args.real_pool_size} clean real rows, found {len(clean_real)}")

    raw_synthetic: list[dict[str, Any]] = []
    synthetic_input_hashes: dict[str, str] = {}
    for path in args.synthetic:
        synthetic_input_hashes[str(path)] = sha256_file(path)
        raw_synthetic.extend(read_jsonl(path))
    synthetic_candidates = [
        row for index, raw in enumerate(raw_synthetic) if (row := to_synthetic_sft(raw, index)) is not None
    ]
    random.Random(args.seed + 1).shuffle(synthetic_candidates)
    clean_synthetic, synthetic_rejected, synthetic_max = collect_clean_rows(
        synthetic_candidates, confirmation_grams, args.similarity_threshold, n_synthetic
    )
    if len(clean_synthetic) < n_synthetic:
        raise SystemExit(f"need {n_synthetic} clean synthetic rows, found {len(clean_synthetic)}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    def completion_budget(row: dict[str, Any]) -> int:
        completion = str(row["completion"])
        if tokenizer is None:
            return len(completion)
        return len(tokenizer.encode(completion, add_special_tokens=False))

    for row in clean_real:
        row["_match_length"] = completion_budget(row)
    for row in clean_synthetic:
        row["_match_length"] = completion_budget(row)

    shared = clean_real[:n_shared_real]
    replacement_pool = clean_real[n_shared_real:]
    matched = match_replacements(replacement_pool, clean_synthetic)

    slots: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in shared:
        slots.append((row, dict(row)))
    for real, synthetic, _ in matched:
        slots.append((real, synthetic))
    order = list(range(len(slots)))
    random.Random(args.seed + 2).shuffle(order)

    control: list[dict[str, Any]] = []
    treatment: list[dict[str, Any]] = []
    for output_index, slot_index in enumerate(order):
        control_row, treatment_row = slots[slot_index]
        slot_id = f"slot-{slot_index:04d}"
        control.append({**strip_internal(control_row), "slot_id": slot_id, "arm": "control"})
        treatment.append({**strip_internal(treatment_row), "slot_id": slot_id, "arm": "treatment"})

    write_jsonl(args.control_output, control)
    write_jsonl(args.treatment_output, treatment)

    matched_diffs = [difference for _, _, difference in matched]
    control_labels = Counter(row["gold"] for row in control)
    treatment_labels = Counter(row["gold"] for row in treatment)
    control_chars = sum(sft_length(row) for row in control)
    treatment_chars = sum(sft_length(row) for row in treatment)
    control_completion_budget = sum(completion_budget(row) for row in control)
    treatment_completion_budget = sum(completion_budget(row) for row in treatment)
    budget_relative_difference = (
        (treatment_completion_budget - control_completion_budget) / control_completion_budget
    )
    audit = {
        "schema_version": "1.0",
        "created_at": "2026-08-28",
        "policy": "nested matched control; source+label+nearest-completion-budget replacement; reject confirmation 3-gram Jaccard >= threshold",
        "seed": args.seed,
        "n_total_per_arm": args.n_total,
        "synthetic_fraction": args.synthetic_fraction,
        "similarity_threshold": args.similarity_threshold,
        "inputs": {
            "real_raw": {"path": str(args.real_raw), "sha256": sha256_file(args.real_raw)},
            "synthetic": synthetic_input_hashes,
            "confirmation": {"path": str(args.confirmation), "sha256": sha256_file(args.confirmation)},
        },
        "outputs": {
            "control": {"path": str(args.control_output), "sha256": sha256_file(args.control_output), "records": len(control)},
            "treatment": {"path": str(args.treatment_output), "sha256": sha256_file(args.treatment_output), "records": len(treatment)},
        },
        "composition": {
            "control": dict(sorted(Counter(row["source"] for row in control).items())),
            "treatment": dict(sorted(Counter(row["source"] for row in treatment).items())),
            "control_label_distribution": dict(sorted(control_labels.items())),
            "treatment_label_distribution": dict(sorted(treatment_labels.items())),
        },
        "matching": {
            "match_unit": "completion_tokens" if tokenizer is not None else "completion_characters",
            "tokenizer": str(args.tokenizer) if args.tokenizer else None,
            "replacement_pairs": len(matched),
            "max_match_unit_difference": max(matched_diffs, default=0),
            "mean_match_unit_difference": sum(matched_diffs) / len(matched_diffs) if matched_diffs else 0.0,
            "control_completion_budget": control_completion_budget,
            "treatment_completion_budget": treatment_completion_budget,
            "completion_budget_relative_difference": budget_relative_difference,
            "control_total_characters": control_chars,
            "treatment_total_characters": treatment_chars,
            "total_character_relative_difference": (treatment_chars - control_chars) / control_chars,
        },
        "filtering": {
            "real_rejected": real_rejected,
            "synthetic_rejected": synthetic_rejected,
            "max_similarity_seen_real": real_max,
            "max_similarity_seen_synthetic": synthetic_max,
        },
        "invariants": {
            "equal_arm_records": len(control) == len(treatment) == args.n_total,
            "equal_slot_ids": {row["slot_id"] for row in control} == {row["slot_id"] for row in treatment},
            "equal_label_distribution": control_labels == treatment_labels,
            "control_prohibited_overlap": sum(row["confirmation_max_3gram_jaccard"] >= args.similarity_threshold for row in control),
            "treatment_prohibited_overlap": sum(row["confirmation_max_3gram_jaccard"] >= args.similarity_threshold for row in treatment),
            "control_unique_questions": len({question_from_prompt(row["prompt"]) for row in control}) == len(control),
            "treatment_unique_questions": len({question_from_prompt(row["prompt"]) for row in treatment}) == len(treatment),
            "completion_budget_relative_difference_le_1pct": abs(budget_relative_difference) <= 0.01,
        },
        "remaining_requirement": "Freeze equal-step, same-batch and completion-only-loss training config in the confirmation manifest.",
    }
    invariant_ok = all(
        value if isinstance(value, bool) else value == 0
        for value in audit["invariants"].values()
    )
    if not invariant_ok:
        raise RuntimeError(f"matched replacement invariant failed: {audit['invariants']}")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
