"""Validate Week 20 inputs, checkpoints, evaluations, and combined summary.

The experiments take many hours, so shell-script completion markers are not a
strong enough completion criterion.  This command performs a fast, read-only
integrity check and exits non-zero when an artifact is missing or inconsistent.
It intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Callable

from artifact_lineage import verify_arms, verify_data_stage


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_SWEEP = REPO_ROOT / "phase1/results/week20_distill"
DEFAULT_SFT = REPO_ROOT / "phase1/results/week19_distill/data/distill_sft.jsonl"

KD_ARMS = ("kd_t2", "kd_t5", "kd_pure")
RS_ARMS = ("rs_mcq", "rs_teacher", "rs_both")
ALL_ARMS = KD_ARMS + RS_ARMS
SCOPES = (
    "logits",
    "samples",
    "scores",
    "prepared",
    "checkpoint",
    "evaluation",
    "feature-results",
    "complete",
)
KD_CONFIGS = {
    "kd_t2": {"alpha": 0.5, "temperature": 2.0},
    "kd_t5": {"alpha": 0.5, "temperature": 5.0},
    "kd_pure": {"alpha": 0.0, "temperature": 2.0},
}
MEDICAL_TASKS = (
    "cmmlu_anatomy", "cmmlu_clinical_knowledge", "cmmlu_college_medicine",
    "cmmlu_professional_medicine", "cmmlu_genetics",
    "cmmlu_traditional_chinese_medicine", "cmmlu_virology", "cmmlu_nutrition",
)
GENERAL_TASKS = (
    "cmmlu_world_history", "cmmlu_high_school_physics", "cmmlu_economics",
    "cmmlu_marxist_theory",
)


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def ok(self, message: str) -> None:
        self.checks.append(message)


def load_json(path: Path, result: Validation) -> dict[str, Any] | None:
    if not path.is_file():
        result.error(f"missing file: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.error(f"invalid JSON {path}: {exc}")
        return None
    if not isinstance(value, dict):
        result.error(f"expected JSON object: {path}")
        return None
    return value


def load_jsonl(path: Path, result: Validation) -> list[dict[str, Any]]:
    if not path.is_file():
        result.error(f"missing file: {path}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.error(f"invalid JSONL {path}:{line_number}: {exc}")
                    continue
                if not isinstance(row, dict):
                    result.error(f"expected object at {path}:{line_number}")
                    continue
                rows.append(row)
    except OSError as exc:
        result.error(f"cannot read {path}: {exc}")
    return rows


def unique_values(
    rows: list[dict[str, Any]],
    key: str | Callable[[dict[str, Any]], Any],
    label: str,
    expected: int,
    result: Validation,
) -> set[Any]:
    get_value = (lambda row: row.get(key)) if isinstance(key, str) else key
    values = [get_value(row) for row in rows]
    missing = sum(value is None for value in values)
    if missing:
        result.error(f"{label}: {missing} records have no identifier")
    present = [value for value in values if value is not None]
    unique = set(present)
    if len(rows) != expected:
        result.error(f"{label}: expected {expected} records, found {len(rows)}")
    if len(unique) != expected:
        result.error(f"{label}: expected {expected} unique identifiers, found {len(unique)}")
    if len(present) != len(unique):
        result.error(f"{label}: found {len(present) - len(unique)} duplicate identifiers")
    if len(rows) == expected and len(unique) == expected and not missing:
        result.ok(f"{label}: {expected} unique records")
    return unique


def compare_sets(left: set[Any], right: set[Any], labels: str, result: Validation) -> None:
    if not left or not right:
        return
    if left != right:
        result.error(
            f"{labels}: identifier sets differ "
            f"(left-only={len(left - right)}, right-only={len(right - left)})"
        )


def close_enough(actual: Any, expected: Any, tolerance: float = 5e-5) -> bool:
    if actual is None or expected is None:
        return actual is expected
    try:
        return math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0)
    except (TypeError, ValueError):
        return False


def resolved_artifact_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def valid_safetensors(path: Path) -> bool:
    """Validate the safetensors header and that tensor offsets fill the file."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return False
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > size - 8:
                return False
            header = json.loads(handle.read(header_length))
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False
    if not isinstance(header, dict):
        return False
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1]
        ):
            return False
        ranges.append(tuple(offsets))
    if not ranges:
        return False
    ranges.sort()
    return ranges[0][0] == 0 and all(
        left[1] == right[0] for left, right in zip(ranges, ranges[1:])
    ) and ranges[-1][1] == size - 8 - header_length


def validate_run_config(
    config: dict[str, Any] | None,
    arm: str,
    sweep: Path,
    source_sft: Path,
    expected_base: Path,
    expected_train: int,
    topk: int,
    result: Validation,
) -> None:
    if config is None:
        return
    error_count = len(result.errors)
    expected_data = source_sft if arm in KD_ARMS else sweep / f"data/{arm}_sft.jsonl"
    checks = {
        "model": (resolved_artifact_path(config.get("model")), expected_base),
        "data": (resolved_artifact_path(config.get("data")), expected_data.resolve()),
        "n_samples": (config.get("n_samples"), expected_train),
        "limit": (config.get("limit"), 0),
        "max_steps": (config.get("max_steps"), 0),
        "lr": (config.get("lr"), 2e-5),
        "epochs": (config.get("epochs"), 3.0),
        "batch_size": (config.get("batch_size"), 4),
        "grad_accum": (config.get("grad_accum"), 4),
        "max_length": (config.get("max_length"), 1536),
        "dtype": (config.get("dtype"), "bfloat16"),
        "seed": (config.get("seed"), 123),
        "lora": (config.get("lora"), {"rank": 16, "alpha": 32, "dropout": 0.05}),
    }
    if arm in KD_ARMS:
        checks.update(
            method=(config.get("method"), "logit_distillation_KD"),
            logits=(
                resolved_artifact_path(config.get("logits")),
                (sweep / "data/teacher_topk_logits.jsonl").resolve(),
            ),
            alpha=(config.get("alpha"), KD_CONFIGS[arm]["alpha"]),
            temperature=(config.get("temperature"), KD_CONFIGS[arm]["temperature"]),
            topk=(config.get("topk"), topk),
        )
    for key, (actual, expected) in checks.items():
        matches = close_enough(actual, expected) if isinstance(expected, float) else actual == expected
        if not matches:
            result.error(f"{arm} run_config.{key}: found {actual!r}, expected {expected!r}")
    if len(result.errors) == error_count:
        result.ok(f"{arm}: run configuration provenance matches the named arm")


def validate_logits(
    sweep: Path,
    source_sft: Path,
    expected_train: int,
    topk: int,
    vocab_size: int,
    result: Validation,
) -> set[Any]:
    source_rows = load_jsonl(source_sft, result)
    source_ids = unique_values(source_rows, "question_id", "source SFT", expected_train, result)
    rows = load_jsonl(sweep / "data/teacher_topk_logits.jsonl", result)
    qids = unique_values(rows, "question_id", "teacher logits", expected_train, result)
    compare_sets(source_ids, qids, "source SFT vs teacher logits", result)

    bad_shape = 0
    for row in rows:
        completion = row.get("completion_token_ids")
        positions = row.get("positions")
        prompt_len = row.get("prompt_len")
        if type(prompt_len) is not int or prompt_len <= 0:
            bad_shape += 1
            continue
        if not isinstance(completion, list) or not isinstance(positions, list) or not completion:
            bad_shape += 1
            continue
        if any(type(token) is not int or not 0 <= token < vocab_size for token in completion):
            bad_shape += 1
            continue
        if len(completion) != len(positions):
            bad_shape += 1
            continue
        for expected_position, position in enumerate(positions):
            tokens = position.get("topk_tokens") if isinstance(position, dict) else None
            logits = position.get("topk_logits") if isinstance(position, dict) else None
            if not isinstance(tokens, list) or not isinstance(logits, list):
                bad_shape += 1
                break
            if len(tokens) != topk or len(logits) != topk:
                bad_shape += 1
                break
            if position.get("pos") != expected_position:
                bad_shape += 1
                break
            if (
                any(type(token) is not int or not 0 <= token < vocab_size for token in tokens)
                or len(set(tokens)) != topk
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in logits
                )
            ):
                bad_shape += 1
                break
    if bad_shape:
        result.error(
            f"teacher logits: {bad_shape} records have invalid lengths, token IDs, or finite top-{topk} values"
        )
    elif rows:
        result.ok(f"teacher logits: completion positions all contain top-{topk} values")
    return qids


def validate_samples(
    sweep: Path, expected_train: int, samples_per_question: int, result: Validation
) -> tuple[set[Any], set[tuple[Any, int]]]:
    rows = load_jsonl(sweep / "data/student_samples.jsonl", result)
    qids = unique_values(rows, "question_id", "student samples", expected_train, result)
    pairs: set[tuple[Any, int]] = set()
    bad = 0
    for row in rows:
        samples = row.get("samples")
        if not isinstance(samples, list) or len(samples) != samples_per_question:
            bad += 1
            continue
        qid = row.get("question_id")
        for index, sample in enumerate(samples):
            pairs.add((qid, index))
            if not isinstance(sample, dict) or not isinstance(sample.get("text"), str):
                bad += 1
                break
            if sample.get("letter") not in (None, "A", "B", "C", "D", "E"):
                bad += 1
                break
            if not isinstance(sample.get("correct"), bool):
                bad += 1
                break
    if bad:
        result.error(f"student samples: {bad} questions contain invalid sample records")
    elif rows:
        result.ok(f"student samples: {samples_per_question} valid samples per question")
    return qids, pairs


def validate_scores(
    sweep: Path, expected_pairs: set[tuple[Any, int]], expected_count: int, result: Validation
) -> set[tuple[Any, int]]:
    rows = load_jsonl(sweep / "data/judge_scores.jsonl", result)
    pairs = unique_values(
        rows,
        lambda row: (row.get("question_id"), row.get("sample_idx"))
        if row.get("question_id") is not None and row.get("sample_idx") is not None
        else None,
        "judge scores",
        expected_count,
        result,
    )
    invalid_scores = sum(
        not isinstance(row.get("score"), int) or not 1 <= row["score"] <= 5 for row in rows
    )
    if invalid_scores:
        result.error(f"judge scores: {invalid_scores} scores are not integers in [1, 5]")
    elif rows:
        result.ok("judge scores: every score is an integer in [1, 5]")
    compare_sets(expected_pairs, pairs, "student samples vs judge scores", result)
    return pairs


def validate_prepared(
    sweep: Path, sample_ids: set[Any], expected_train: int, result: Validation
) -> None:
    for arm in RS_ARMS:
        rows = load_jsonl(sweep / f"data/{arm}_sft.jsonl", result)
        qids = unique_values(rows, "question_id", f"{arm} SFT", expected_train, result)
        compare_sets(sample_ids, qids, f"student samples vs {arm} SFT", result)
        invalid_source = sum(row.get("source") not in ("student", "teacher_fallback") for row in rows)
        if invalid_source:
            result.error(f"{arm} SFT: {invalid_source} invalid source values")
    dpo_rows = load_jsonl(sweep / "data/dpo_onpolicy.jsonl", result)
    dpo_ids = unique_values(
        dpo_rows, "question_id", "on-policy DPO", len(dpo_rows), result
    )
    if not dpo_rows:
        result.error("on-policy DPO: no usable preference pairs")
    if not dpo_ids.issubset(sample_ids):
        result.error(
            f"on-policy DPO: {len(dpo_ids - sample_ids)} question IDs are absent from student samples"
        )
    malformed = sum(
        any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("prompt", "chosen", "rejected"))
        for row in dpo_rows
    )
    if malformed:
        result.error(f"on-policy DPO: {malformed} pairs have missing or empty text fields")
    if dpo_rows:
        result.ok(
            f"on-policy DPO: {len(dpo_rows)}/{expected_train} questions have distinct usable pairs"
        )
    identical = sum(
        isinstance(row.get("chosen"), str)
        and isinstance(row.get("rejected"), str)
        and row["chosen"].strip() == row["rejected"].strip()
        for row in dpo_rows
    )
    if identical:
        result.error(f"on-policy DPO: {identical} chosen/rejected pairs are identical")


def validate_checkpoint(
    sweep: Path,
    arm: str,
    expected_train: int,
    topk: int,
    source_sft: Path,
    expected_base: Path,
    result: Validation,
) -> None:
    adapter = sweep / arm / "adapter_model.safetensors"
    if not adapter.is_file() or not valid_safetensors(adapter):
        result.error(f"missing, truncated, or invalid adapter checkpoint: {adapter}")
    config = load_json(sweep / arm / "run_config.json", result)
    validate_run_config(
        config,
        arm,
        sweep,
        source_sft,
        expected_base,
        expected_train,
        topk,
        result,
    )


def validate_evaluation(
    sweep: Path, arm: str, expected_eval: int, result: Validation
) -> dict[str, Any] | None:
    error_count = len(result.errors)
    fused = sweep / f"{arm}_fused"
    load_json(fused / "config.json", result)
    fused_weights = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in fused.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not fused_weights:
        result.error(f"missing fused model weights: {fused}")
    elif any(path.suffix == ".safetensors" and not valid_safetensors(path) for path in fused_weights):
        result.error(f"truncated or invalid fused safetensors weights: {fused}")
    evaluation = load_json(fused / "cmexam_holdout.json", result)
    if not evaluation:
        return None
    accuracy = evaluation.get("accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool) or not 0.0 <= accuracy <= 1.0:
        result.error(f"{arm}: invalid CMExam accuracy {accuracy!r}")
    if evaluation.get("n") != expected_eval:
        result.error(f"{arm}: expected CMExam n={expected_eval}, found {evaluation.get('n')!r}")
    if resolved_artifact_path(evaluation.get("model")) != fused.resolve():
        result.error(
            f"{arm}: CMExam model provenance is {evaluation.get('model')!r}, expected {fused}"
        )
    predictions = load_jsonl(fused / "cmexam_holdout.json.preds.jsonl", result)
    # eval_cmexam.py intentionally persists only the first 50 generations as
    # a human-readable sanity sample; the aggregate JSON carries n=500.
    expected_prediction_sample = min(50, expected_eval)
    if len(predictions) != expected_prediction_sample:
        result.error(
            f"{arm}: expected {expected_prediction_sample} sampled CMExam predictions, "
            f"found {len(predictions)}"
        )
    if len(result.errors) == error_count:
        result.ok(f"{arm}: fused model and CMExam evaluation are complete")
    return evaluation


def validate_results(
    sweep: Path,
    arms: tuple[str, ...],
    expected_eval: int,
    expected_train: int,
    topk: int,
    source_sft: Path,
    expected_base: Path,
    result: Validation,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    domain = load_json(sweep / "domain_gain.json", result) or {}
    forgetting = load_json(sweep / "forgetting.json", result) or {}
    domain_runs = domain.get("runs") if isinstance(domain.get("runs"), dict) else {}
    forgetting_runs = forgetting.get("runs") if isinstance(forgetting.get("runs"), dict) else {}
    for label, value in (
        ("domain_gain base_medical_cn", domain.get("base_medical_cn")),
        ("forgetting base_general_cn", forgetting.get("base_general_cn")),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            result.error(f"{label}: expected a finite number, found {value!r}")
    evaluations: dict[str, dict[str, Any]] = {}
    base_scores = load_json(
        sweep / "base_hf" / f"scores_{expected_base.name}.json", result
    ) or {}

    def mean_score(scores: dict[str, Any], tasks: tuple[str, ...], label: str) -> float | None:
        values = [scores.get(task) for task in tasks]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ):
            result.error(f"{label}: missing or non-finite raw CMMLU task scores")
            return None
        return round(sum(values) / len(values), 4)

    raw_base_med = mean_score(base_scores, MEDICAL_TASKS, "base medical CMMLU")
    raw_base_gen = mean_score(base_scores, GENERAL_TASKS, "base general CMMLU")
    if not close_enough(domain.get("base_medical_cn"), raw_base_med):
        result.error("domain_gain.json base does not reconcile with raw base scores")
    if not close_enough(forgetting.get("base_general_cn"), raw_base_gen):
        result.error("forgetting.json base does not reconcile with raw base scores")

    for arm in arms:
        if arm not in domain_runs:
            result.error(f"domain_gain.json: missing arm {arm}")
        elif (
            not isinstance(domain_runs[arm], (int, float))
            or isinstance(domain_runs[arm], bool)
            or not math.isfinite(domain_runs[arm])
        ):
            result.error(f"domain_gain.json: {arm} must be finite, found {domain_runs[arm]!r}")
        if arm not in forgetting_runs:
            result.error(f"forgetting.json: missing arm {arm}")
        elif (
            not isinstance(forgetting_runs[arm], (int, float))
            or isinstance(forgetting_runs[arm], bool)
            or not math.isfinite(forgetting_runs[arm])
        ):
            result.error(f"forgetting.json: {arm} must be finite, found {forgetting_runs[arm]!r}")
        raw_scores = load_json(sweep / arm / f"scores_{arm}_fused.json", result) or {}
        raw_med = mean_score(raw_scores, MEDICAL_TASKS, f"{arm} medical CMMLU")
        raw_gen = mean_score(raw_scores, GENERAL_TASKS, f"{arm} general CMMLU")
        expected_med_delta = round(raw_med - raw_base_med, 4) if raw_med is not None and raw_base_med is not None else None
        expected_gen_delta = round(raw_gen - raw_base_gen, 4) if raw_gen is not None and raw_base_gen is not None else None
        if not close_enough(domain_runs.get(arm), expected_med_delta):
            result.error(f"domain_gain.json: {arm} does not reconcile with raw scores")
        if not close_enough(forgetting_runs.get(arm), expected_gen_delta):
            result.error(f"forgetting.json: {arm} does not reconcile with raw scores")
        validate_checkpoint(
            sweep,
            arm,
            expected_train,
            topk,
            source_sft,
            expected_base,
            result,
        )
        evaluation = validate_evaluation(sweep, arm, expected_eval, result)
        if not evaluation:
            continue
        evaluations[arm] = evaluation
    if all(arm in evaluations for arm in arms):
        result.ok(f"results: checkpoints and {expected_eval}-example evaluations exist for {len(arms)} arms")
    return domain, forgetting, evaluations


def validate_summary(
    sweep: Path,
    domain: dict[str, Any],
    forgetting: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    expected_train: int,
    result: Validation,
) -> None:
    summary = load_json(sweep / "week20_summary.json", result)
    if not summary:
        return
    if summary.get("n_train") != expected_train:
        result.error(f"summary: expected n_train={expected_train}, found {summary.get('n_train')!r}")
    arm_rows = summary.get("arms")
    if not isinstance(arm_rows, list):
        result.error("summary: arms must be a list")
        return
    arm_map = {row.get("variant"): row for row in arm_rows if isinstance(row, dict)}
    if set(arm_map) != set(ALL_ARMS):
        result.error(
            f"summary: expected arms {sorted(ALL_ARMS)}, found {sorted(str(key) for key in arm_map)}"
        )
    base = summary.get("base_metrics") if isinstance(summary.get("base_metrics"), dict) else {}
    base_cmexam = base.get("cmexam_holdout")
    base_medical = domain.get("base_medical_cn")
    base_general = forgetting.get("base_general_cn")
    domain_runs = domain.get("runs") if isinstance(domain.get("runs"), dict) else {}
    forgetting_runs = forgetting.get("runs") if isinstance(forgetting.get("runs"), dict) else {}

    for arm in ALL_ARMS:
        row = arm_map.get(arm)
        evaluation = evaluations.get(arm)
        if not row or not evaluation:
            continue
        expected_values = {
            "cmexam_holdout": evaluation.get("accuracy"),
            "cmexam_delta": round(evaluation["accuracy"] - base_cmexam, 4)
            if isinstance(evaluation.get("accuracy"), (int, float))
            and isinstance(base_cmexam, (int, float))
            else None,
            "cmmlu_medical": round(base_medical + domain_runs[arm], 4)
            if isinstance(base_medical, (int, float)) and isinstance(domain_runs.get(arm), (int, float))
            else None,
            "cmmlu_medical_delta": domain_runs.get(arm),
            "cmmlu_general": round(base_general + forgetting_runs[arm], 4)
            if isinstance(base_general, (int, float))
            and isinstance(forgetting_runs.get(arm), (int, float))
            else None,
            "cmmlu_general_delta": forgetting_runs.get(arm),
            "cmexam_n": evaluation.get("n"),
        }
        for key, expected in expected_values.items():
            if not close_enough(row.get(key), expected):
                result.error(
                    f"summary {arm}.{key}: found {row.get(key)!r}, expected {expected!r}"
                )
    if not result.errors:
        result.ok("summary: all six arms reconcile with raw evaluation artifacts")


def run_validation(args: argparse.Namespace) -> Validation:
    result = Validation()
    sweep = Path(args.sweep).expanduser().resolve()
    source_sft = Path(args.source_sft).expanduser().resolve()
    expected_base = Path(args.expected_base).expanduser().resolve()
    teacher = Path(args.teacher).expanduser().resolve()

    selected_arms = tuple(args.arms or ALL_ARMS)
    invalid_arms = sorted(set(selected_arms) - set(ALL_ARMS))
    if invalid_arms:
        result.error(f"unknown arms: {invalid_arms}")
        return result

    if args.scope == "checkpoint":
        for arm in selected_arms:
            validate_checkpoint(
                sweep,
                arm,
                args.expected_train,
                args.topk,
                source_sft,
                expected_base,
                result,
            )
        if not args.skip_lineage:
            result.errors.extend(
                verify_arms(
                    Path(args.lineage_manifest).resolve(), sweep, expected_base,
                    source_sft, list(selected_arms), "checkpoint"
                )
            )
            stages = set()
            if any(arm in KD_ARMS for arm in selected_arms):
                stages.add("logits")
            if any(arm in RS_ARMS for arm in selected_arms):
                stages.update(("samples", "scores"))
            for stage in stages:
                result.errors.extend(
                    verify_data_stage(
                        Path(args.lineage_manifest).resolve(), sweep, expected_base,
                        source_sft, teacher, stage
                    )
                )
        return result

    if args.scope == "evaluation":
        for arm in selected_arms:
            validate_evaluation(sweep, arm, args.expected_eval, result)
        if not args.skip_lineage:
            result.errors.extend(
                verify_arms(
                    Path(args.lineage_manifest).resolve(), sweep, expected_base,
                    source_sft, list(selected_arms), "evaluation"
                )
            )
            stages = set()
            if any(arm in KD_ARMS for arm in selected_arms):
                stages.add("logits")
            if any(arm in RS_ARMS for arm in selected_arms):
                stages.update(("samples", "scores"))
            for stage in stages:
                result.errors.extend(
                    verify_data_stage(
                        Path(args.lineage_manifest).resolve(), sweep, expected_base,
                        source_sft, teacher, stage
                    )
                )
        return result

    if args.scope == "logits":
        validate_logits(
            sweep, source_sft, args.expected_train, args.topk, args.vocab_size, result
        )
        return result

    if args.scope == "feature-results":
        validate_logits(
            sweep, source_sft, args.expected_train, args.topk, args.vocab_size, result
        )
        validate_results(
            sweep,
            KD_ARMS,
            args.expected_eval,
            args.expected_train,
            args.topk,
            source_sft,
            expected_base,
            result,
        )
        if not args.skip_lineage:
            result.errors.extend(
                verify_arms(
                    Path(args.lineage_manifest).resolve(), sweep, expected_base,
                    source_sft, list(KD_ARMS), "evaluation"
                )
            )
            result.errors.extend(
                verify_data_stage(
                    Path(args.lineage_manifest).resolve(), sweep, expected_base,
                    source_sft, teacher, "logits"
                )
            )
        return result

    sample_ids, sample_pairs = validate_samples(
        sweep, args.expected_train, args.samples_per_question, result
    )
    if args.scope == "samples":
        return result

    validate_scores(
        sweep,
        sample_pairs,
        args.expected_train * args.samples_per_question,
        result,
    )
    if args.scope == "scores":
        return result

    validate_prepared(sweep, sample_ids, args.expected_train, result)
    if args.scope == "prepared":
        return result

    domain, forgetting, evaluations = validate_results(
        sweep,
        ALL_ARMS,
        args.expected_eval,
        args.expected_train,
        args.topk,
        source_sft,
        expected_base,
        result,
    )
    if args.scope == "complete":
        logits_ids = validate_logits(
            sweep,
            source_sft,
            args.expected_train,
            args.topk,
            args.vocab_size,
            result,
        )
        compare_sets(logits_ids, sample_ids, "teacher logits vs student samples", result)
        validate_summary(
            sweep, domain, forgetting, evaluations, args.expected_train, result
        )
        if not args.skip_lineage:
            result.errors.extend(
                verify_arms(
                    Path(args.lineage_manifest).resolve(), sweep, expected_base,
                    source_sft, list(ALL_ARMS), "evaluation"
                )
            )
            for stage in ("samples", "scores", "logits"):
                result.errors.extend(
                    verify_data_stage(
                        Path(args.lineage_manifest).resolve(), sweep, expected_base,
                        source_sft, teacher, stage
                    )
                )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Week 20 experiment completion")
    parser.add_argument("--sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--source-sft", default=str(DEFAULT_SFT))
    parser.add_argument(
        "--expected-base",
        default=str(REPO_ROOT / "phase1/results/week12_lora_cpt/50_50_fused"),
    )
    parser.add_argument("--scope", choices=SCOPES, default="complete")
    parser.add_argument("--arms", nargs="*", choices=ALL_ARMS)
    parser.add_argument("--expected-train", type=int, default=2000)
    parser.add_argument("--samples-per-question", type=int, default=8)
    parser.add_argument("--expected-eval", type=int, default=500)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--lineage-manifest", default=str(HERE / "week20_lineage.json"))
    parser.add_argument(
        "--teacher",
        default=str(Path.home() / ".lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"),
    )
    parser.add_argument("--skip-lineage", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_validation(args)
    for check in result.checks:
        print(f"[ok] {check}")
    if result.errors:
        for error in result.errors:
            print(f"[error] {error}", file=sys.stderr)
        print(f"[FAILED] Week 20 validation found {len(result.errors)} error(s)", file=sys.stderr)
        return 1
    print(f"[PASS] Week 20 validation ({args.scope}) completed with {len(result.checks)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
