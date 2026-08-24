"""Compute reproducible synthetic-data quality metrics without heavy dependencies."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from common import jaccard, mcq_content_sha256, ngrams, normalize_mcq, read_jsonl, write_jsonl


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text.lower())


def distinct_n(rows: list[dict[str, Any]], n: int) -> float:
    all_ngrams = []
    for row in rows:
        values = tokens(row["question"])
        all_ngrams.extend(tuple(values[i : i + n]) for i in range(max(0, len(values) - n + 1)))
    return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0.0


def approximate_self_bleu(rows: list[dict[str, Any]], sample_size: int, seed: int) -> float:
    """Nearest-neighbour bigram overlap; a dependency-free Self-BLEU proxy."""
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    selected = selected[:sample_size]
    grams = [ngrams(row["question"], n=2) for row in selected]
    scores = []
    for index, own in enumerate(grams):
        scores.append(max((jaccard(own, other) for j, other in enumerate(grams) if j != index), default=0.0))
    return statistics.fmean(scores) if scores else 0.0


def js_divergence(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left) | set(right))
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return 0.0
    p = [left[key] / left_total for key in keys]
    q = [right[key] / right_total for key in keys]
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0 and y > 0)

    return (kl(p, m) + kl(q, m)) / 2


def normalized_rows(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    out = []
    errors: Counter[str] = Counter()
    for row in read_jsonl(path):
        normalized, error = normalize_mcq(row)
        if error:
            errors[error] += 1
        else:
            normalized.update({key: row.get(key) for key in ("synthetic_id", "method", "backend", "evolution_depth")})
            out.append(normalized)
    return out, errors


def nearest_real_similarity(synthetic: list[dict[str, Any]], real: list[dict[str, Any]]) -> list[float]:
    real_grams = [ngrams(row["question"]) for row in real]
    return [max((jaccard(ngrams(row["question"]), other) for other in real_grams), default=0.0) for row in synthetic]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-data", required=True, type=Path)
    parser.add_argument("--evolved-data", type=Path)
    parser.add_argument("--real-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--audit-size", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--real-sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    self_rows, self_errors = normalized_rows(args.self_data)
    evolved_rows, evolved_errors = (normalized_rows(args.evolved_data) if args.evolved_data else ([], Counter()))
    real_rows_all, real_errors = normalized_rows(args.real_data)
    real_rows = list(real_rows_all)
    random.Random(args.seed).shuffle(real_rows)
    real_rows = real_rows[: args.real_sample_size]
    synthetic = self_rows + evolved_rows
    if not synthetic or not real_rows:
        raise SystemExit("synthetic and real inputs must both contain valid records")

    similarities = nearest_real_similarity(synthetic, real_rows)
    questions = [row["question"] for row in synthetic]
    exact_duplicates = len(questions) - len(set(questions))
    lengths_syn = [len(row["question"]) for row in synthetic]
    lengths_real = [len(row["question"]) for row in real_rows]
    explanation_support = []
    for row in synthetic:
        correct = next(item["value"] for item in row["options"] if item["key"] == row["answer"])
        signal = correct[: min(6, len(correct))]
        explanation_support.append(bool(signal and signal in row["explanation"]))

    previous_audit = {}
    if args.audit_output.is_file():
        previous_audit = {
            row.get("synthetic_id"): row for row in read_jsonl(args.audit_output)
            if row.get("synthetic_id")
        }
    rng = random.Random(args.seed)
    audit_pool = list(synthetic)
    rng.shuffle(audit_pool)
    audit = []
    for row in audit_pool[: min(args.audit_size, len(audit_pool))]:
        prior = previous_audit.get(row.get("synthetic_id"), {})
        content_sha256 = mcq_content_sha256(row)
        if prior.get("content_sha256") != content_sha256 or (
            prior and prior.get("content_sha256") != mcq_content_sha256(prior)
        ):
            prior = {}
        audit.append({
            "synthetic_id": row.get("synthetic_id"), "question": row["question"],
            "options": row["options"], "answer": row["answer"], "explanation": row["explanation"],
            "content_sha256": content_sha256,
            "review_correct": prior.get("review_correct"),
            "review_notes": prior.get("review_notes", ""),
            **({"reviewer": prior["reviewer"]} if prior.get("reviewer") else {}),
            **({"reviewer_kind": prior["reviewer_kind"]} if prior.get("reviewer_kind") else {}),
        })
    write_jsonl(args.audit_output, audit)
    reviewed = [row for row in audit if isinstance(row.get("review_correct"), bool)]
    review_accuracy = sum(row["review_correct"] for row in reviewed) / len(reviewed) if reviewed else None
    review_kinds = sorted({str(row.get("reviewer_kind")) for row in reviewed if row.get("reviewer_kind")})

    metrics = {
        "methodology": {
            "self_bleu": "dependency-free nearest-neighbour character-bigram Jaccard proxy; lower means more diverse",
            "correctness": "structural validity and answer-text support are automated proxies; the separate manual review was performed by an AI, not a clinician or human annotator",
            "distribution": "question length and answer-label Jensen-Shannon divergence",
        },
        "counts": {
            "self": len(self_rows), "evolved": len(evolved_rows), "synthetic_total": len(synthetic),
            "real_total": len(real_rows_all), "real_comparison_sample": len(real_rows),
        },
        "validity": {
            "synthetic_valid_rate": round(len(synthetic) / max(len(synthetic) + sum(self_errors.values()) + sum(evolved_errors.values()), 1), 6),
            "self_errors": dict(self_errors), "evolved_errors": dict(evolved_errors), "real_errors": dict(real_errors),
            "answer_explanation_support_rate": round(sum(explanation_support) / len(explanation_support), 6),
        },
        "diversity": {
            "exact_duplicate_count": exact_duplicates,
            "exact_duplicate_rate": round(exact_duplicates / len(synthetic), 6),
            "distinct_1": round(distinct_n(synthetic, 1), 6),
            "distinct_2": round(distinct_n(synthetic, 2), 6),
            "approx_self_bleu": round(approximate_self_bleu(synthetic, args.sample_size, args.seed), 6),
        },
        "novelty_vs_real": {
            "mean_max_3gram_jaccard": round(statistics.fmean(similarities), 6),
            "p95_max_3gram_jaccard": round(sorted(similarities)[min(len(similarities) - 1, int(0.95 * len(similarities)))], 6),
            "exact_real_match_count": sum(value == 1.0 for value in similarities),
        },
        "distribution_shift": {
            "mean_question_chars_synthetic": round(statistics.fmean(lengths_syn), 3),
            "mean_question_chars_real": round(statistics.fmean(lengths_real), 3),
            "mean_length_ratio": round(statistics.fmean(lengths_syn) / statistics.fmean(lengths_real), 6),
            "answer_label_js_divergence": round(js_divergence(Counter(row["answer"] for row in synthetic), Counter(row["answer"] for row in real_rows)), 6),
            "synthetic_answer_distribution": dict(sorted(Counter(row["answer"] for row in synthetic).items())),
            "real_answer_distribution": dict(sorted(Counter(row["answer"] for row in real_rows).items())),
        },
        "evolution": {
            "count": len(evolved_rows),
            "depth_distribution": dict(sorted(Counter(str(row.get("evolution_depth")) for row in evolved_rows).items())),
        },
        "manual_review": {
            "path": str(args.audit_output), "sample_size": len(audit), "reviewed": len(reviewed),
            "accuracy": round(review_accuracy, 6) if review_accuracy is not None else None,
            "reviewer_kinds": review_kinds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[quality] metrics -> {args.output}; audit sample -> {args.audit_output}")


if __name__ == "__main__":
    main()
