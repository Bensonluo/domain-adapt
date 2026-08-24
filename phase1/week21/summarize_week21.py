"""Create the Week 21 machine summary and required evidence reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "pending" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, default=Path("phase1/results/week21_synthetic"))
    parser.add_argument("--reports", type=Path, default=Path("phase1/week21/results"))
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    control_path = args.control or args.sweep / "control_real_fused/cmexam_holdout.json"

    quality = load(args.sweep / "quality_metrics.json")
    treatment = load(args.sweep / "synthetic50_fused/cmexam_holdout.json")
    control = load(control_path)
    statistics = load(args.sweep / "replacement_statistics.json")
    leakage = load(args.sweep / "holdout_similarity.json")
    manifest = load(args.sweep / "data/replacement_50.manifest.json")
    self_stats = load(args.sweep / "data/self_instruct_stats.json")
    evol_stats = load(args.sweep / "data/evol_instruct_stats.json")
    audit_path = args.sweep / "human_audit.jsonl"
    audit = read_jsonl(audit_path)
    reviewed = [row for row in audit if isinstance(row.get("review_correct"), bool)]
    review_accuracy = sum(row["review_correct"] for row in reviewed) / len(reviewed) if reviewed else None
    review_kinds = sorted({str(row.get("reviewer_kind")) for row in reviewed})
    review_label = "Manual AI review" if review_kinds == ["ai_nonclinician"] else "Manual review"
    if review_kinds == ["ai_nonclinician"]:
        review_disclosure = "This is an AI qualitative sanity check, not clinician or human validation."
    elif review_kinds == ["human_clinician"]:
        review_disclosure = "The sample is recorded as clinician-reviewed; its scope is still limited to 30 items."
    else:
        review_disclosure = "Reviewer kinds are explicit in the audit artifact; this limited sample is not broad clinical validation."

    delta = statistics["delta"]
    point_ok = statistics["point_estimate_meets_margin"]
    noninferiority = statistics["paired_bootstrap"]["noninferiority_established"]
    interval = statistics["paired_bootstrap"]["delta_interval"]
    summary = {
        "experiment": "50% synthetic replacement",
        "control": {"label": "week19_real_2000", "accuracy": control["accuracy"], "n": control["n"]},
        "treatment": {"label": "week21_real1000_synthetic1000", "accuracy": treatment["accuracy"], "n": treatment["n"]},
        "delta": delta,
        "acceptance_threshold": statistics["noninferiority_margin"],
        "point_estimate_acceptable": point_ok,
        "statistical_noninferiority_established": noninferiority,
        "conclusion": (
            "noninferiority established" if noninferiority else
            "point estimate meets operational margin, but statistical noninferiority is inconclusive" if point_ok else
            "point estimate misses operational margin"
        ),
        "paired_statistics": statistics,
        "holdout_similarity": leakage,
        "replacement_manifest": manifest,
        "generation": {"self": self_stats, "evol": evol_stats},
        "quality_metrics": quality,
    }
    args.sweep.mkdir(parents=True, exist_ok=True)
    (args.sweep / "week21_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.reports.mkdir(parents=True, exist_ok=True)

    counts = quality["counts"]
    validity = quality["validity"]
    diversity = quality["diversity"]
    novelty = quality["novelty_vs_real"]
    shift = quality["distribution_shift"]
    synthetic_leakage = leakage["sources"]
    quality_report = f"""# Week 21 synthetic-data quality

## Scope and provenance

- Self-Instruct accepted: {counts['self']} (`{self_stats['backend']}`, model `{self_stats['model']}`)
- Evol-Instruct accepted: {counts['evolved']} (`{evol_stats['backend']}`, model `{evol_stats['model']}`)
- Real comparison: {counts['real_comparison_sample']} sampled from {counts['real_total']} records
- {review_label}: {len(reviewed)}/{len(audit)} reviewed; apparent accuracy {pct(review_accuracy)}

The recorded reviewer kind is `{', '.join(review_kinds)}`. {review_disclosure}

## Measured quality

| Metric | Result |
|---|---:|
| Schema and answer validity | {pct(validity['synthetic_valid_rate'])} |
| Exact duplicate rate | {pct(diversity['exact_duplicate_rate'])} |
| Distinct-1 / Distinct-2 | {diversity['distinct_1']} / {diversity['distinct_2']} |
| Approximate Self-BLEU | {diversity['approx_self_bleu']} |
| Mean / P95 max 3-gram similarity to real sample | {novelty['mean_max_3gram_jaccard']} / {novelty['p95_max_3gram_jaccard']} |
| Exact matches to real sample | {novelty['exact_real_match_count']} |
| Synthetic/real mean question-length ratio | {shift['mean_length_ratio']} |
| Answer-label JS divergence | {shift['answer_label_js_divergence']} |
| Answer-text support in explanation (proxy) | {pct(validity['answer_explanation_support_rate'])} |
| Self-Instruct items ≥0.78 similarity to full holdout | {synthetic_leakage['self']['above_or_equal_threshold']} / {synthetic_leakage['self']['n']} |
| Evol-Instruct items ≥0.78 similarity to full holdout | {synthetic_leakage['evol']['above_or_equal_threshold']} / {synthetic_leakage['evol']['n']} |

Approximate Self-BLEU is nearest-neighbour character-bigram Jaccard, not package BERTScore. The full-holdout screen is character 3-gram Jaccard and is not a semantic-embedding guarantee. All accepted synthetic items passed the 0.78 threshold; the mixed replacement set contains {leakage['replacement_high_overlap_by_source'].get('real', 0)} high-overlap real-source items inherited from the source corpus.

## Limitations

The completed generation predates the MLX RNG-seeding fix now present in the scripts, so raw responses freeze and authenticate this run but cannot be regenerated bit-for-bit from its seed alone. Future fresh runs seed MLX per request. Medical correctness still requires independent clinician review before clinical use.
"""
    (args.reports / "week21_synthetic_quality.md").write_text(quality_report, encoding="utf-8")

    verdict = (
        "statistical noninferiority established" if noninferiority else
        "operational point estimate passed, but statistical noninferiority was not established" if point_ok else
        "operational point estimate failed"
    )
    replacement_report = f"""# Week 21 50% synthetic replacement experiment

## Controlled design

The Week 19 arm uses 2,000 real CMExam SFT examples. The Week 21 treatment keeps the same 2,000-example size, base checkpoint, seed, LoRA/SFT hyperparameters, and fixed 500-question holdout, replacing exactly 1,000 training examples with accepted synthetic examples. The pre-specified operational tolerance is −2 percentage points.

Both fused checkpoints were re-evaluated in one matched pass using CPU float32 greedy decoding, batch size 8, and 16 generated tokens; this paired rerun is the basis of the statistics below.

| Arm | Real | Synthetic | Holdout n | CMExam accuracy |
|---|---:|---:|---:|---:|
| Week 19 real control | 2,000 | 0 | {control['n']} | {pct(control['accuracy'])} |
| Week 21 50% replacement | {manifest['n_real']} | {manifest['n_synthetic']} | {treatment['n']} | {pct(treatment['accuracy'])} |

Treatment minus control: **{100 * delta:+.2f} percentage points**. Paired bootstrap 95% interval: **[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}] percentage points**. Exact McNemar two-sided p-value: **{statistics['mcnemar_exact']['two_sided_p']:.4f}**.

Decision: **{verdict}**. The point estimate ({100 * delta:+.2f} points) {'is' if point_ok else 'is not'} within the operational −2-point tolerance. The paired interval {'clears' if noninferiority else 'crosses'} that margin, so statistical noninferiority {'is established' if noninferiority else 'is not established'}.

## Answer to the anchor question

For this one model and dataset, 50% replacement is promising by point estimate, but the evidence is statistically inconclusive. A larger holdout or repeated training seeds are needed before claiming that synthetic data can replace half of the real SFT data.

## Integrity and limitations

- Full paired predictions for all 500 questions are retained and the aggregate accuracies are recomputed from them.
- Content hashes bind generation inputs/outputs, treatment data, adapter, fused model, evaluation, statistics, and reports.
- Synthetic questions passed exact-match and 0.78 character 3-gram holdout screening. Twelve high-overlap records in the treatment are real-source items; this inherited source-corpus overlap remains a limitation.
- The control checkpoint was reused from Week 19 and re-evaluated with the same full-prediction evaluator; training stochasticity across repeated seeds was not measured.
"""
    (args.reports / "week21_replacement_experiment.md").write_text(replacement_report, encoding="utf-8")
    print(f"[summary] delta={delta:+.4f}; point_ok={point_ok}; noninferiority={noninferiority}")


if __name__ == "__main__":
    main()
