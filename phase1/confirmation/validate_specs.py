#!/usr/bin/env python3
"""Validate frozen Phase 1 confirmation contracts before expensive runs."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = Path(__file__).resolve().parent
SPEC_FILES = ("cpt.json", "dpo_ipo.json", "grpo.json", "distillation.json", "synthetic_replacement.json")
EXPECTED_METHODS = {"cpt", "dpo_ipo", "grpo", "distillation", "synthetic_replacement"}
FORBIDDEN_CONFIRMATION_PATHS = {
    "phase1/data/processed/cmexam/holdout.jsonl",
    "phase1/data/processed/cmexam/test.jsonl",
    "phase1/data/processed/preference/holdout.jsonl",
    "phase1/data/processed/preference/train_split.jsonl",
}
DEVELOPMENT_HASHES = {
    "cmmlu_local_all": "7eb87616fa446bee45a8c0754f3c1faf3b6ca33f35ac740311d25013ebd889ba",
    "cmexam_test_and_historical_500": "bbeed1d863bcd573448e93f6b77fcc4b57f93ea2e08bcc98de94544af7417377",
}
EXPECTED_CONFIRMATION_PATH = (REPO_ROOT / "phase1/data/processed/cmexam/confirmation_candidate_v1.jsonl").resolve()
EXPECTED_CONFIRMATION_HASH = "c6d5f692dff505853345b026bb60b842a68ee83b9d47fe9b9ef2e4128aae5e8e"


class SpecError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_file(relative: str, location: str) -> Path:
    require(isinstance(relative, str) and relative, f"{location}: invalid path")
    require(not Path(relative).is_absolute(), f"{location}: absolute paths are forbidden")
    root = REPO_ROOT.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"{location}: path escapes repository")
    require(path.is_file(), f"{location}: missing {relative}")
    return path


def require_hashed_file(value: Any, location: str) -> Path:
    require(isinstance(value, dict), f"{location}: expected object")
    require(isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
            f"{location}: missing or invalid sha256")
    path = resolve_repo_file(value.get("path"), location)
    require(sha256_file(path) == value["sha256"], f"{location}: hash drift for {value['path']}")
    return path


def validate_hashed_paths(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        if "path" in value:
            require_hashed_file(value, location)
        for key, child in value.items():
            validate_hashed_paths(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_hashed_paths(child, f"{location}[{index}]")


def validate_spec(spec: dict[str, Any]) -> None:
    prefix = str(spec.get("spec_id", "unnamed"))
    require(spec.get("schema_version") == "1.0", f"{prefix}: unsupported schema")
    require(spec.get("method") in EXPECTED_METHODS, f"{prefix}: unknown method")
    require(spec.get("status") == "FROZEN_BEFORE_EXECUTION", f"{prefix}: plan is not frozen")

    seeds = spec.get("seeds", [])
    require(len(seeds) >= 3 and len(set(seeds)) == len(seeds), f"{prefix}: need at least 3 unique seeds")
    require(all(isinstance(seed, int) for seed in seeds), f"{prefix}: seeds must be integers")

    base_model = spec.get("base_model", {})
    require(base_model.get("id"), f"{prefix}: missing base model id")
    require(re.fullmatch(r"[0-9a-f]{40}", str(base_model.get("revision", ""))) is not None,
            f"{prefix}: base model revision must be an immutable commit")
    recipe = spec.get("training_recipe", {})
    require(len(recipe) >= 6, f"{prefix}: training recipe is not executable")
    require(recipe.get("optimizer"), f"{prefix}: missing optimizer")
    require(recipe.get("learning_rate", 0) > 0, f"{prefix}: missing learning rate")
    require(recipe.get("stopping_rule"), f"{prefix}: missing stopping rule")

    claim = spec.get("claim", {})
    for field in ("hypothesis", "interpretation_if_supported", "wording_if_rejected", "wording_if_inconclusive"):
        require(isinstance(claim.get(field), str) and claim[field].strip(), f"{prefix}: missing claim.{field}")
    comparison = spec.get("comparison", {})
    if spec["method"] == "cpt":
        estimands = comparison.get("estimands", [])
        require(len(estimands) == 2, f"{prefix}: CPT requires exposure and adaptation-mode estimands")
        require({item.get("id") for item in estimands} == {"cpt_exposure", "adaptation_mode"},
                f"{prefix}: CPT estimands are not isolated")
        for item in estimands:
            require(item.get("control") and item.get("treatment"), f"{prefix}: incomplete CPT estimand")
            require(item.get("treatment_variable") == item.get("id"), f"{prefix}: CPT treatment variable mismatch")
            require(len(item.get("fixed_controls", [])) >= 4, f"{prefix}: insufficient CPT fixed controls")
    else:
        require(comparison.get("control"), f"{prefix}: missing control")
        require(comparison.get("treatments"), f"{prefix}: missing treatment")
        require(comparison.get("treatment_variable"), f"{prefix}: treatment variable not isolated")
        require(len(comparison.get("fixed_controls", [])) >= 4, f"{prefix}: insufficient fixed controls")

    data = spec.get("data", {})
    training = data.get("training")
    require(isinstance(training, list) and training, f"{prefix}: missing training data")
    for index, item in enumerate(training):
        require_hashed_file(item, f"{prefix}.data.training[{index}]")
    development = data.get("development")
    require(isinstance(development, dict), f"{prefix}: missing development data")
    require(isinstance(development.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", development["sha256"]),
            f"{prefix}: development data lacks hash")
    if "path" in development:
        require_hashed_file(development, f"{prefix}.data.development")
    else:
        require(development.get("id") in DEVELOPMENT_HASHES, f"{prefix}: unknown development benchmark id")
        require(development["sha256"] == DEVELOPMENT_HASHES[development["id"]],
                f"{prefix}: development benchmark hash drift")
    confirmation = data.get("confirmation", {})
    confirmation_path = resolve_repo_file(confirmation.get("path"), f"{prefix}.data.confirmation")
    forbidden_paths = {(REPO_ROOT / path).resolve() for path in FORBIDDEN_CONFIRMATION_PATHS}
    require(confirmation_path not in forbidden_paths,
            f"{prefix}: reused historical holdout as confirmation")
    require_hashed_file(confirmation, f"{prefix}.data.confirmation")
    require(confirmation_path == EXPECTED_CONFIRMATION_PATH and confirmation["sha256"] == EXPECTED_CONFIRMATION_HASH,
            f"{prefix}: must use the frozen confirmation candidate")
    require(confirmation.get("status") == "FROZEN_CONFIRMATION_CANDIDATE_NOT_BLIND",
            f"{prefix}: wrong confirmation status")
    require(confirmation.get("is_blind") is False, f"{prefix}: local candidate must not be called blind")

    analysis = spec.get("analysis", {})
    require(analysis.get("primary_metric"), f"{prefix}: missing primary metric")
    require(analysis.get("paired_bootstrap_samples", 0) >= 10000, f"{prefix}: insufficient bootstrap samples")
    require(analysis.get("mcnemar") is True, f"{prefix}: McNemar test is required")
    require(analysis.get("decision_rule"), f"{prefix}: missing decision rule")
    require(analysis.get("multiple_comparison_policy"), f"{prefix}: missing multiplicity policy")

    external = spec.get("external_blind", {})
    require(external.get("candidate_is_external_blind") is False, f"{prefix}: candidate mislabeled external blind")

    method = spec["method"]
    details = spec.get("method_specific", {})
    if method != "cpt":
        weights = base_model.get("weights")
        weights_path = require_hashed_file(weights, f"{prefix}.base_model.weights")
        require(weights_path.suffix == ".safetensors", f"{prefix}: starting weights must be a safetensors file")
        require(weights.get("size_bytes") == weights_path.stat().st_size and weights_path.stat().st_size >= 100_000_000,
                f"{prefix}: starting weights size is missing or implausible")
    if method == "cpt":
        require(details.get("verify_trainable_parameter_names_before_run") is True,
                f"{prefix}: full/LoRA parameter scope must be verified")
        require(details.get("optimization_token_budget", 0) > 0, f"{prefix}: missing optimization-token budget")
    elif method == "dpo_ipo":
        training_paths = {item.get("path") for item in data["training"]}
        require("phase1/data/processed/preference/train_grouped_v1.jsonl" in training_paths,
                f"{prefix}: grouped preference v1 is required")
        require(details.get("grouped_prompt_overlap_required") == 0, f"{prefix}: grouped overlap must be zero")
        require(details.get("matched_sft_baseline_required") is True, f"{prefix}: matched SFT baseline required")
        require(details.get("independent_outcome_metric") == "cmexam_accuracy",
                f"{prefix}: outcome must be independent of preference objective")
    elif method == "grpo":
        require(details.get("required_training_confirmation_overlap") == 0,
                f"{prefix}: GRPO confirmation overlap must be zero")
        protocol = details.get("probe_fixture", {})
        generator_path = require_hashed_file(protocol.get("generator"), f"{prefix}.method_specific.probe_fixture.generator")
        fixture_path = require_hashed_file(protocol.get("fixture"), f"{prefix}.method_specific.probe_fixture.fixture")
        audit_path = require_hashed_file(protocol.get("audit"), f"{prefix}.method_specific.probe_fixture.audit")
        require(generator_path.suffix == ".py" and fixture_path.suffix == ".jsonl" and audit_path.suffix == ".json",
                f"{prefix}: invalid probe fixture artifact types")
        require(protocol.get("selection_algorithm") == "sort immutable confirmation id within answer label; take fixed prefix",
                f"{prefix}: probe selection algorithm drift")
        require(protocol.get("selection_seed") == "NOT_APPLICABLE_NO_RNG",
                f"{prefix}: probe selection seed policy drift")
        fixture_rows = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        fixture_counts: dict[str, int] = {}
        for row in fixture_rows:
            probe_id = str(row.get("probe_id"))
            fixture_counts[probe_id] = fixture_counts.get(probe_id, 0) + 1
        expected_counts = {
            "label_prior_only": 500, "format_only": 100, "empty_explanation": 100,
            "incorrect_explanation_with_correct_label": 100, "unparseable_output": 1,
        }
        require(fixture_counts == expected_counts, f"{prefix}: probe fixture counts drift")
        probes = details.get("reward_hacking_probes", [])
        require({probe.get("id") for probe in probes} == {
            "label_prior_only", "format_only", "empty_explanation",
            "incorrect_explanation_with_correct_label", "unparseable_output",
        }, f"{prefix}: reward-hacking probes incomplete")
        for probe in probes:
            probe_id = probe.get("id")
            fixture_filter = probe.get("fixture_filter", {})
            require(fixture_filter.get("probe_id") == probe_id and fixture_filter.get("expected_records") == expected_counts[probe_id],
                    f"{prefix}: probe fixture filter incomplete")
            require(isinstance(probe.get("metric"), str) and len(probe["metric"]) >= 5,
                    f"{prefix}: probe metric incomplete")
            threshold = probe.get("pass_threshold")
            require(isinstance(threshold, dict) and threshold.get("operator") == "lte"
                    and isinstance(threshold.get("value"), (int, float)) and not isinstance(threshold.get("value"), bool),
                    f"{prefix}: probe threshold invalid")
    elif method == "distillation":
        require(details.get("alpha_values") == [1.0, 0.5, 0.0], f"{prefix}: alpha arms not preregistered")
        require(details.get("same_prompt_completion_ids_required") is True,
                f"{prefix}: hard/soft KD must use the same examples")
        require(details.get("equal_optimization_tokens_required") is True,
                f"{prefix}: hard/soft KD token budgets must match")
    elif method == "synthetic_replacement":
        require(details.get("noninferiority_margin") == -0.02, f"{prefix}: noninferiority margin must be frozen")
        require(details.get("required_training_confirmation_overlap") == 0,
                f"{prefix}: synthetic training/confirmation overlap must be zero")
        estimator = analysis.get("hierarchical_bootstrap", {})
        require(estimator.get("outer_unit") == "training_seed", f"{prefix}: bootstrap must include seed variation")
        require(estimator.get("inner_unit") == "paired_confirmation_item",
                f"{prefix}: bootstrap must preserve item pairing")
        require(estimator.get("replicates") >= 10000, f"{prefix}: hierarchical bootstrap too small")
        require(estimator.get("decision_ci_lower_bound_gt") == -0.02,
                f"{prefix}: hierarchical decision margin drift")

    validate_hashed_paths(data, prefix + ".data")
    validate_hashed_paths(base_model, prefix + ".base_model")


def load_and_validate_all(spec_dir: Path = SPEC_DIR) -> list[dict[str, Any]]:
    specs = [json.loads((spec_dir / filename).read_text(encoding="utf-8")) for filename in SPEC_FILES]
    require({spec.get("method") for spec in specs} == EXPECTED_METHODS, "method coverage mismatch")
    for spec in specs:
        validate_spec(spec)
    return specs


def main() -> None:
    specs = load_and_validate_all()
    print(f"PASS: {len(specs)} frozen Phase 1 confirmation specifications are valid")


if __name__ == "__main__":
    main()
