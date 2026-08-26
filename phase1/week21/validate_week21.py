"""Deep integrity validation for Week 21 evidence and experiment lineage."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path

from analyze_replacement import exact_mcnemar_p
from artifact_lineage import STAGES, current_stage, read_manifest
from common import mcq_content_sha256, normalize_mcq, prompt_from_mcq, read_jsonl, stable_id
from evol_instruct import accepted_record
from self_instruct import replay_raw

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / "phase1").is_dir())


class Result:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.ok: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        (self.ok if condition else self.errors).append(message)


def load_json(path: Path, result: Result) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("root is not an object")
        result.ok.append(f"valid JSON: {path.name}")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.errors.append(f"invalid or missing JSON {path}: {exc}")
        return {}


def rows(path: Path, result: Result) -> list[dict]:
    try:
        return read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.errors.append(f"invalid or missing JSONL {path}: {exc}")
        return []


def valid_safetensors(path: Path) -> bool:
    item_sizes = {
        "BOOL": 1, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
    }
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return False
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 2 <= header_length <= 100 * 1024 * 1024:
                return False
            header = json.loads(handle.read(header_length))
        if not isinstance(header, dict):
            return False
        spans = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, dict) or tensor.get("dtype") not in item_sizes:
                return False
            shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
            if not isinstance(shape, list) or not all(type(size) is int and size >= 0 for size in shape):
                return False
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(value) is int for value in offsets):
                return False
            start, end = offsets
            elements = math.prod(shape)
            if start < 0 or end < start or end - start != elements * item_sizes[tensor["dtype"]]:
                return False
            spans.append((start, end))
        if not spans:
            return False
        spans.sort()
        if spans[0][0] != 0 or any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
            return False
        return path.stat().st_size == 8 + header_length + spans[-1][1]
    except (OSError, ValueError, struct.error, json.JSONDecodeError):
        return False


def finite_rate(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def declared_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def check_code(result: Result) -> None:
    required = (
        "common.py", "prepare_seeds.py", "self_instruct.py", "evol_instruct.py",
        "assess_quality.py", "apply_audit_labels.py", "prepare_replacement.py",
        "check_holdout_similarity.py", "merge_adapter.py", "eval_cmexam_full.py",
        "analyze_replacement.py", "artifact_lineage.py", "summarize_week21.py", "run_week21.sh",
    )
    for name in required:
        path = HERE / name
        result.require(path.is_file(), f"code exists: {name}")
        if path.is_file():
            result.require("TODO: Implement" not in path.read_text(encoding="utf-8"), f"no stub: {name}")


def check_generation(sweep: Path, teacher: Path, n_self: int, n_evol: int, result: Result) -> None:
    data = sweep / "data"
    seeds = rows(data / "seed_instructions.jsonl", result)
    self_raw = rows(data / "self_instruct_raw.jsonl", result)
    self_rows = rows(data / "self_instruct.jsonl", result)
    self_stats = load_json(data / "self_instruct_stats.json", result)
    result.require(len(seeds) == 100, "exactly 100 unique seeds")
    result.require(len({row.get("seed_id") for row in seeds}) == len(seeds), "seed IDs unique")
    result.require(all(normalize_mcq(row)[1] is None for row in seeds), "all seeds valid MCQs")
    train = rows(REPO_ROOT / "phase1/data/processed/cmexam/train.jsonl", result)
    test = rows(REPO_ROOT / "phase1/data/processed/cmexam/test.jsonl", result)
    holdout_questions = {str(row.get("Question", "")).strip() for row in test}
    seed_pool = []
    for raw in train:
        normalized, error = normalize_mcq(raw)
        if error or normalized["question"] in holdout_questions:
            continue
        normalized["seed_id"] = f"seed-{stable_id(normalized['question'])}"
        normalized["source"] = "CMExam/train"
        seed_pool.append(normalized)
    random.Random(123).shuffle(seed_pool)
    result.require(seeds == seed_pool[:100], "seed set recomputes exactly from fixed train/holdout sources")
    expected_teacher = str(teacher.resolve())
    result.require(bool(self_raw) and all(row.get("backend") == "mlx" for row in self_raw), "Self raw uses MLX only")
    result.require(bool(self_raw) and all(str(Path(str(row.get("model"))).resolve()) == expected_teacher for row in self_raw), "Self raw binds expected teacher")
    normalized_seeds = []
    for seed in seeds:
        normalized, error = normalize_mcq(seed)
        if not error:
            normalized["seed_id"] = seed["seed_id"]
            normalized_seeds.append(normalized)
    replayed, _ = replay_raw(self_raw, normalized_seeds, 0.78, n_self)
    result.require(self_rows == replayed, "Self accepted records replay exactly from raw responses")
    result.require(len(self_rows) == n_self, f"exactly {n_self} Self records")
    result.require(len({row.get("synthetic_id") for row in self_rows}) == len(self_rows), "Self IDs unique")
    result.require(self_stats.get("accepted") == len(self_rows) and self_stats.get("raw_generations") == len(self_raw), "Self stats reconcile")
    seeded = all("rng_seed" in row for row in self_raw)
    result.require(self_stats.get("mlx_rng_seeded", False) is seeded, "Self RNG reproducibility flag is honest")

    evol_raw = rows(data / "evol_instruct_raw.jsonl", result)
    evol_rows = rows(data / "evol_instruct.jsonl", result)
    evol_stats = load_json(data / "evol_instruct_stats.json", result)
    self_by_id = {row["synthetic_id"]: row for row in self_rows}
    raw_by_source: dict[str, list[dict]] = {}
    for raw in evol_raw:
        raw_by_source.setdefault(str(raw.get("source_id")), []).append(raw)
    result.require(bool(evol_raw) and all(row.get("backend") == "mlx" for row in evol_raw), "Evol raw uses MLX only")
    result.require(bool(evol_raw) and all(str(Path(str(row.get("model"))).resolve()) == expected_teacher for row in evol_raw), "Evol raw binds expected teacher")
    linked = True
    for evolved in evol_rows:
        source = self_by_id.get(str(evolved.get("source_id")))
        if source is None:
            linked = False
            continue
        candidates = [accepted_record(raw, source, 0.98)[0] for raw in raw_by_source.get(evolved["source_id"], [])]
        if evolved not in candidates:
            linked = False
    result.require(linked, "every Evol record links exactly to raw response and Self source")
    result.require(len(evol_rows) == n_evol, f"exactly {n_evol} Evol records")
    result.require(all(row.get("evolution_depth") == len(row.get("stages", [])) in (1, 2, 3) for row in evol_rows), "Evol stages/depth valid")
    result.require(evol_stats.get("accepted") == len(evol_rows), "Evol stats reconcile")
    seeded = all("rng_seed" in row for row in evol_raw)
    result.require(evol_stats.get("mlx_rng_seeded", False) is seeded, "Evol RNG reproducibility flag is honest")


def check_audit(sweep: Path, result: Result) -> None:
    audit = rows(sweep / "human_audit.jsonl", result)
    ids = [row.get("synthetic_id") for row in audit]
    result.require(len(ids) == 30 and len(ids) == len(set(ids)), "manual AI review has fixed 30 unique records")
    result.require(all(row.get("content_sha256") == mcq_content_sha256(row) for row in audit), "review labels bind full medically material content")
    result.require(all(isinstance(row.get("review_correct"), bool) and str(row.get("review_notes", "")).strip() for row in audit), "manual review labels and notes complete")
    allowed_kinds = {"ai_nonclinician", "human_nonclinician", "human_clinician"}
    result.require(all(row.get("reviewer_kind") in allowed_kinds and str(row.get("reviewer", "")).strip() for row in audit), "reviewer identity and kind are explicit")
    quality = load_json(sweep / "quality_metrics.json", result)
    review = quality.get("manual_review", {})
    review_kinds = sorted({row["reviewer_kind"] for row in audit})
    result.require(review.get("reviewed") == 30 and review.get("reviewer_kinds") == review_kinds, "quality metrics label reviewers honestly")
    expected_accuracy = sum(row["review_correct"] for row in audit) / len(audit)
    result.require(
        type(review.get("accuracy")) in (int, float)
        and math.isclose(review["accuracy"], expected_accuracy, abs_tol=5e-7),
        "review accuracy reconciles",
    )
    result.require(declared_path(review.get("path")) == (sweep / "human_audit.jsonl").resolve(), "quality review path binds sweep audit")


def check_replacement(sweep: Path, result: Result) -> None:
    data = sweep / "data"
    replacement = rows(data / "replacement_50.jsonl", result)
    manifest = load_json(data / "replacement_50.manifest.json", result)
    self_rows = rows(data / "self_instruct.jsonl", result)
    evol_rows = rows(data / "evol_instruct.jsonl", result)
    real_rows = rows(REPO_ROOT / "phase1/results/week19_distill/data/real_sft.jsonl", result)
    real_by_id = {f"real-{row.get('question_id', index)}": row for index, row in enumerate(real_rows)}
    synthetic_by_id = {row["synthetic_id"]: row for row in self_rows + evol_rows}
    reconciled = True
    for row in replacement:
        if row.get("source") == "real":
            source = real_by_id.get(str(row.get("replacement_id")))
            if source is None or any(row.get(key) != source.get(key) for key in ("prompt", "completion", "question_id", "gold")):
                reconciled = False
        elif row.get("source") == "synthetic":
            source = synthetic_by_id.get(str(row.get("replacement_id")))
            if source is None or row.get("prompt") != prompt_from_mcq(source) or row.get("completion") != f"{source['answer']}\n{source['explanation']}" or row.get("gold") != source["answer"]:
                reconciled = False
        else:
            reconciled = False
    real = sum(row.get("source") == "real" for row in replacement)
    synthetic = sum(row.get("source") == "synthetic" for row in replacement)
    result.require(len(replacement) == 2000 and real == synthetic == 1000, "replacement is 1,000 real + 1,000 synthetic")
    result.require(len({row.get("replacement_id") for row in replacement}) == 2000, "replacement IDs unique")
    result.require(reconciled, "replacement rows reconcile exactly to source artifacts")
    result.require(manifest.get("n_total") == 2000 and manifest.get("n_real") == real and manifest.get("n_synthetic") == synthetic, "replacement manifest reconciles")
    leakage = load_json(sweep / "holdout_similarity.json", result)
    sources = leakage.get("sources", {})
    result.require(sources.get("self", {}).get("above_or_equal_threshold") == 0 and sources.get("evol", {}).get("above_or_equal_threshold") == 0, "accepted synthetic data passes full-holdout 3-gram screen")


def check_checkpoint(sweep: Path, base: Path, result: Result) -> None:
    adapter = sweep / "synthetic50"
    weights = adapter / "adapter_model.safetensors"
    result.require(valid_safetensors(weights), "adapter safetensors header and payload valid")
    config = load_json(adapter / "run_config.json", result)
    expected_data = (sweep / "data/replacement_50.jsonl").resolve()
    result.require(Path(str(config.get("model", ""))).resolve() == base.resolve(), "adapter run config binds expected base")
    result.require(Path(str(config.get("data", ""))).resolve() == expected_data, "adapter run config binds treatment data")
    expected = {"lr": 2e-5, "epochs": 3.0, "batch_size": 4, "grad_accum": 4, "max_length": 1536, "seed": 123, "n_samples": 2000}
    result.require(all(config.get(key) == value for key, value in expected.items()), "adapter hyperparameters match controlled design")
    result.require(config.get("lora") == {"rank": 16, "alpha": 32, "dropout": 0.05}, "treatment LoRA configuration matches design")
    result.require((adapter / "loss_log.csv").is_file() and len((adapter / "loss_log.csv").read_text().splitlines()) > 2, "training loss log present")
    control_config = load_json(REPO_ROOT / "phase1/results/week19_distill/real/run_config.json", result)
    result.require(declared_path(control_config.get("model")) == base.resolve(), "control run config binds expected base")
    result.require(
        declared_path(control_config.get("data")) == (REPO_ROOT / "phase1/results/week19_distill/data/real_sft.jsonl").resolve(),
        "control run config binds real training data",
    )
    controlled_keys = tuple(expected) + ("lora", "dtype", "completion_only_loss", "packing")
    result.require(all(control_config.get(key) == config.get(key) for key in controlled_keys), "control/treatment training hyperparameters match")


def check_eval_file(
    path: Path, result: Result, label: str, expected_model: Path, expected_data: Path,
    expected_golds: list[str],
) -> tuple[dict, list[dict]]:
    aggregate = load_json(path, result)
    predictions = rows(Path(str(path) + ".preds.jsonl"), result)
    correct = sum(row.get("correct") is True for row in predictions)
    unparseable = sum(row.get("pred") is None for row in predictions)
    result.require(len(predictions) == 500 and [row.get("index") for row in predictions] == list(range(500)), f"{label} retains all 500 ordered predictions")
    result.require(all(isinstance(row.get("correct"), bool) and row.get("correct") == (row.get("pred") == row.get("gold")) for row in predictions), f"{label} correctness fields recompute")
    result.require(aggregate.get("n") == len(predictions) and aggregate.get("correct") == correct, f"{label} counts reconcile")
    result.require(finite_rate(aggregate.get("accuracy")) and aggregate.get("accuracy") == correct / max(len(predictions), 1), f"{label} accuracy reconciles")
    result.require(aggregate.get("unparseable") == unparseable and aggregate.get("unparseable_rate") == unparseable / max(len(predictions), 1), f"{label} unparseable reconciles")
    result.require(declared_path(aggregate.get("model")) == expected_model.resolve(), f"{label} aggregate binds expected model")
    result.require(declared_path(aggregate.get("data")) == expected_data.resolve(), f"{label} aggregate binds fixed holdout")
    result.require([str(row.get("gold")) for row in predictions] == expected_golds, f"{label} gold labels match fixed holdout")
    result.require(
        aggregate.get("decoding") == "greedy" and aggregate.get("max_new_tokens") == 16
        and aggregate.get("batch_size") == 8 and aggregate.get("device") == "cpu"
        and aggregate.get("dtype") == "torch.float32",
        f"{label} uses fixed CPU/float32 greedy evaluation",
    )
    return aggregate, predictions


def check_evaluation(sweep: Path, control_model: Path, result: Result) -> None:
    fused = sweep / "synthetic50_fused/model.safetensors"
    result.require(valid_safetensors(fused), "fused safetensors header and payload valid")
    holdout_path = REPO_ROOT / "phase1/data/processed/cmexam/holdout.jsonl"
    holdout = rows(holdout_path, result)
    golds = [str(row.get("answer", "")).strip().upper() for row in holdout]
    treatment, treatment_preds = check_eval_file(
        sweep / "synthetic50_fused/cmexam_holdout.json", result, "treatment",
        sweep / "synthetic50_fused", holdout_path, golds,
    )
    control, control_preds = check_eval_file(
        sweep / "control_real_fused/cmexam_holdout.json", result, "control",
        control_model, holdout_path, golds,
    )
    result.require(all(left.get("gold") == right.get("gold") for left, right in zip(control_preds, treatment_preds)), "control/treatment predictions pair on identical gold labels")
    statistics = load_json(sweep / "replacement_statistics.json", result)
    delta = treatment.get("accuracy", 0) - control.get("accuracy", 0)
    result.require(
        statistics.get("n_pairs") == 500 and type(statistics.get("delta")) in (int, float)
        and math.isclose(statistics["delta"], delta, abs_tol=1e-12),
        "paired statistics reconcile to full predictions",
    )
    result.require(statistics.get("point_estimate_meets_margin") == (delta >= -0.02), "point-estimate decision recomputes")
    interval = statistics.get("paired_bootstrap", {}).get("delta_interval", [])
    result.require(len(interval) == 2 and interval[0] <= delta <= interval[1], "paired bootstrap interval contains observed delta")
    result.require(statistics.get("paired_bootstrap", {}).get("noninferiority_established") == (bool(interval) and interval[0] > -0.02), "noninferiority decision uses interval lower bound")
    differences = [int(right["correct"]) - int(left["correct"]) for left, right in zip(control_preds, treatment_preds)]
    rng = random.Random(123)
    bootstrap = sorted(
        sum(differences[rng.randrange(500)] for _ in differences) / 500 for _ in range(20_000)
    )
    expected_interval = [bootstrap[500], bootstrap[19_499]]
    control_only = sum(left["correct"] and not right["correct"] for left, right in zip(control_preds, treatment_preds))
    treatment_only = sum(not left["correct"] and right["correct"] for left, right in zip(control_preds, treatment_preds))
    expected_mcnemar = {
        "control_only_correct": control_only, "treatment_only_correct": treatment_only,
        "discordant": control_only + treatment_only,
        "two_sided_p": exact_mcnemar_p(control_only, treatment_only),
    }
    result.require(
        statistics.get("noninferiority_margin") == -0.02
        and statistics.get("paired_bootstrap", {}).get("samples") == 20_000
        and statistics.get("paired_bootstrap", {}).get("seed") == 123
        and statistics.get("paired_bootstrap", {}).get("confidence_level") == 0.95
        and interval == expected_interval,
        "paired bootstrap is recomputed exactly from predictions",
    )
    result.require(statistics.get("mcnemar_exact") == expected_mcnemar, "McNemar statistics recompute exactly")


def check_summary(sweep: Path, reports: Path, result: Result) -> None:
    summary = load_json(sweep / "week21_summary.json", result)
    statistics = load_json(sweep / "replacement_statistics.json", result)
    result.require(summary.get("delta") == statistics.get("delta"), "summary delta reconciles")
    result.require(summary.get("point_estimate_acceptable") == statistics.get("point_estimate_meets_margin"), "summary point decision reconciles")
    result.require(summary.get("statistical_noninferiority_established") == statistics.get("paired_bootstrap", {}).get("noninferiority_established"), "summary inference reconciles")
    for name in ("week21_synthetic_quality.md", "week21_replacement_experiment.md"):
        path = reports / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        result.require(bool(text) and "pending" not in text, f"complete report: {name}")
        result.require("Human correctness audit" not in text and "人工正确性抽检" not in text, f"report avoids false human-review claim: {name}")


def check_lineage(args: argparse.Namespace, result: Result, stages: tuple[str, ...]) -> None:
    try:
        manifest = read_manifest(args.lineage)
        for stage in stages:
            expected = manifest.get("stages", {}).get(stage)
            actual = current_stage(stage, args.sweep, args.base, args.teacher, args.control_model, args.n_self, args.n_evol)
            result.require(expected == actual, f"content lineage verified: {stage}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result.errors.append(f"lineage validation failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("code", "data", "audit", "checkpoint", "evaluation", "complete"), default="complete")
    parser.add_argument("--sweep", type=Path, default=REPO_ROOT / "phase1/results/week21_synthetic")
    parser.add_argument("--reports", type=Path, default=HERE / "results")
    parser.add_argument("--base", type=Path, default=REPO_ROOT / "phase1/results/week12_lora_cpt/50_50_fused")
    parser.add_argument("--teacher", type=Path, default=Path("/Users/luopeng/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"))
    parser.add_argument("--control-model", type=Path, default=REPO_ROOT / "phase1/results/week19_distill/real_fused")
    parser.add_argument("--lineage", type=Path)
    parser.add_argument("--n-self", type=int, default=1000)
    parser.add_argument("--n-evol", type=int, default=250)
    args = parser.parse_args()
    args.sweep = args.sweep.resolve()
    args.reports = args.reports.resolve()
    args.base = args.base.resolve()
    args.teacher = args.teacher.resolve()
    args.control_model = args.control_model.resolve()
    args.lineage = (args.lineage or args.sweep / "artifact_lineage.json").resolve()
    result = Result()
    check_code(result)
    if args.scope in ("data", "audit", "checkpoint", "evaluation", "complete"):
        check_generation(args.sweep, args.teacher, args.n_self, args.n_evol, result)
    if args.scope in ("audit", "checkpoint", "evaluation", "complete"):
        check_audit(args.sweep, result)
    if args.scope in ("checkpoint", "evaluation", "complete"):
        check_replacement(args.sweep, result)
    if args.scope in ("checkpoint", "evaluation", "complete"):
        check_checkpoint(args.sweep, args.base, result)
    if args.scope in ("evaluation", "complete"):
        check_evaluation(args.sweep, args.control_model, result)
    if args.scope == "complete":
        check_summary(args.sweep, args.reports, result)
        check_lineage(args, result, STAGES)
    for message in result.ok:
        print(f"[ok] {message}")
    for message in result.errors:
        print(f"[error] {message}", file=sys.stderr)
    print(f"[validate] {len(result.ok)} passed, {len(result.errors)} failed")
    raise SystemExit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
