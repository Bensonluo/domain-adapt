"""Strict, dependency-light configuration for AdaptStack experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .core.immutable import plain_value

TRAINING_STAGES = ("cpt", "sft", "dpo", "grpo", "distill")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ConfigError(ValueError):
    """Raised when a configuration violates the public schema."""


class UniqueKeySafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return dict(value)


def _unknown(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    extras = sorted(set(mapping) - allowed)
    if extras:
        raise ConfigError(f"{label} contains unknown keys: {', '.join(extras)}")


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, label: str) -> str:
    identifier = _non_empty_string(value, label)
    if IDENTIFIER_PATTERN.fullmatch(identifier) is None:
        raise ConfigError(
            f"{label} must be a safe identifier using letters, digits, '.', '_' or '-'"
        )
    return identifier


def _relative_output_dir(value: Any, label: str) -> str:
    text = _non_empty_string(value, label)
    path = Path(text)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ConfigError(f"{label} must be a relative path without parent traversal")
    return str(path)


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ConfigError(f"{label} must be a non-empty list of strings")
    return tuple(value)


def _validate_bool(value: Any, label: str) -> None:
    if type(value) is not bool:
        raise ConfigError(f"{label} must be boolean")


def _validate_string(value: Any, label: str) -> None:
    _non_empty_string(value, label)


def _validate_string_list(value: Any, label: str) -> None:
    _string_list(value, label)


def _validate_positive_int(value: Any, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")


def _validate_non_negative_int(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{label} must be a non-negative integer")


def _validate_positive_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{label} must be a positive number")


def _validate_probability(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ConfigError(f"{label} must be between 0 and 1")


def _freeze_value(value: Any, label: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{label} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return tuple(_freeze_value(item, f"{label}[]") for item in value)
    if isinstance(value, dict):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ConfigError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze_value(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    raise ConfigError(f"{label} contains unsupported value type {type(value).__name__}")


def _frozen_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    frozen = _freeze_value(dict(value), label)
    if not isinstance(frozen, Mapping):  # pragma: no cover - protected by dict(value)
        raise ConfigError(f"{label} must be a mapping")
    return frozen


Validator = Callable[[Any, str], None]

DATA_SECTION_VALIDATORS: Mapping[str, Mapping[str, Validator]] = {
    "clean": {
        "min_length": _validate_positive_int,
        "dedup_method": _validate_string,
        "dedup_threshold": _validate_probability,
    },
    "sft": {
        "target_examples": _validate_positive_int,
        "format": _validate_string,
    },
    "preference": {"target_pairs": _validate_positive_int},
    "synthetic": {
        "enabled": _validate_bool,
        "provenance_required": _validate_bool,
    },
}

STAGE_OPTION_VALIDATORS: Mapping[str, Mapping[str, Validator]] = {
    "cpt": {
        "learning_rate": _validate_positive_number,
        "epochs": _validate_positive_int,
        "max_steps": _validate_positive_int,
        "warmup_steps": _validate_non_negative_int,
    },
    "sft": {
        "learning_rate": _validate_positive_number,
        "epochs": _validate_positive_int,
        "lora_rank": _validate_positive_int,
        "lora_alpha": _validate_positive_int,
    },
    "dpo": {
        "learning_rate": _validate_positive_number,
        "beta": _validate_positive_number,
    },
    "grpo": {
        "learning_rate": _validate_positive_number,
        "num_generations": _validate_positive_int,
        "reward_functions": _validate_string_list,
    },
    "distill": {
        "teacher_model": _validate_string,
        "temperature": _validate_positive_number,
        "alpha": _validate_probability,
    },
}

EVALUATION_SECTION_VALIDATORS: Mapping[str, Mapping[str, Validator]] = {
    "llm_judge": {
        "enabled": _validate_bool,
        "rubric_version": _validate_string,
    },
    "human_eval": {
        "enabled": _validate_bool,
        "required_reviewers": _validate_positive_int,
    },
}


def _validated_section(
    value: Any,
    label: str,
    validators: Mapping[str, Validator],
) -> Mapping[str, Any]:
    raw = _object(value, label)
    backend_options = raw.pop("backend_options", None)
    _unknown(raw, set(validators), label)
    for key, item in raw.items():
        validators[key](item, f"{label}.{key}")
    if backend_options is not None:
        raw["backend_options"] = _object(backend_options, f"{label}.backend_options")
    return _frozen_mapping(raw, label)


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    domain: str
    output_dir: str = "artifacts/runs"
    seed: int = 123

    @classmethod
    def from_mapping(cls, value: Any) -> ProjectConfig:
        raw = _object(value, "project")
        _unknown(raw, {"name", "domain", "output_dir", "seed"}, "project")
        seed = raw.get("seed", 123)
        if type(seed) is not int or seed < 0:
            raise ConfigError("project.seed must be a non-negative integer")
        return cls(
            name=_non_empty_string(raw.get("name"), "project.name"),
            domain=_identifier(raw.get("domain"), "project.domain"),
            output_dir=_relative_output_dir(
                raw.get("output_dir", "artifacts/runs"), "project.output_dir"
            ),
            seed=seed,
        )


@dataclass(frozen=True)
class DataConfig:
    corpus_path: str
    general_ratio: float
    clean: Mapping[str, Any] = field(default_factory=dict)
    sft: Mapping[str, Any] = field(default_factory=dict)
    preference: Mapping[str, Any] = field(default_factory=dict)
    synthetic: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any) -> DataConfig:
        raw = _object(value, "data")
        allowed = {"corpus_path", "general_ratio", "clean", "sft", "preference", "synthetic"}
        _unknown(raw, allowed, "data")
        ratio = raw.get("general_ratio", 0.0)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
            raise ConfigError("data.general_ratio must be between 0 and 1")
        sections = {}
        for name in ("clean", "sft", "preference", "synthetic"):
            label = f"data.{name}"
            sections[name] = _validated_section(
                raw.get(name, {}), label, DATA_SECTION_VALIDATORS[name]
            )
        return cls(
            corpus_path=_non_empty_string(raw.get("corpus_path"), "data.corpus_path"),
            general_ratio=float(ratio),
            **sections,
        )


@dataclass(frozen=True)
class StageConfig:
    enabled: bool
    backend: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any, label: str) -> StageConfig:
        raw = _object(value, label)
        enabled = raw.pop("enabled", False)
        backend = raw.pop("backend", "unconfigured")
        backend_options = raw.pop("backend_options", None)
        if type(enabled) is not bool:
            raise ConfigError(f"{label}.enabled must be boolean")
        stage_name = label.rsplit(".", maxsplit=1)[-1]
        validators = STAGE_OPTION_VALIDATORS[stage_name]
        _unknown(raw, set(validators), label)
        for key, item in raw.items():
            validators[key](item, f"{label}.{key}")
        if backend_options is not None:
            raw["backend_options"] = _object(backend_options, f"{label}.backend_options")
        return cls(
            enabled=enabled,
            backend=_non_empty_string(backend, f"{label}.backend"),
            options=_frozen_mapping(raw, label),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool
    benchmarks: tuple[str, ...]
    llm_judge: Mapping[str, Any] = field(default_factory=dict)
    human_eval: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any) -> EvaluationConfig:
        raw = _object(value, "evaluation")
        _unknown(raw, {"enabled", "benchmarks", "llm_judge", "human_eval"}, "evaluation")
        enabled = raw.get("enabled", True)
        if type(enabled) is not bool:
            raise ConfigError("evaluation.enabled must be boolean")
        benchmarks = _string_list(raw.get("benchmarks", []), "evaluation.benchmarks")
        return cls(
            enabled=enabled,
            benchmarks=benchmarks,
            llm_judge=_validated_section(
                raw.get("llm_judge", {}),
                "evaluation.llm_judge",
                EVALUATION_SECTION_VALIDATORS["llm_judge"],
            ),
            human_eval=_validated_section(
                raw.get("human_eval", {}),
                "evaluation.human_eval",
                EVALUATION_SECTION_VALIDATORS["human_eval"],
            ),
        )


@dataclass(frozen=True)
class TrackingConfig:
    backend: str = "local"
    project: str = "adaptstack"
    mode: str = "offline"

    @classmethod
    def from_mapping(cls, value: Any) -> TrackingConfig:
        raw = _object(value, "tracking")
        _unknown(raw, {"backend", "project", "mode"}, "tracking")
        backend = _non_empty_string(raw.get("backend", "local"), "tracking.backend")
        if backend not in {"local", "wandb"}:
            raise ConfigError("tracking.backend must be 'local' or 'wandb'")
        mode = _non_empty_string(raw.get("mode", "offline"), "tracking.mode")
        if mode not in {"offline", "online", "disabled"}:
            raise ConfigError("tracking.mode must be offline, online, or disabled")
        return cls(
            backend=backend,
            project=_non_empty_string(raw.get("project", "adaptstack"), "tracking.project"),
            mode=mode,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    base_model: str
    data: DataConfig
    training: Mapping[str, StageConfig]
    evaluation: EvaluationConfig
    tracking: TrackingConfig

    @classmethod
    def from_mapping(cls, value: Any) -> ExperimentConfig:
        raw = _object(value, "config")
        required = {"project", "base_model", "data", "training", "evaluation", "tracking"}
        _unknown(raw, required, "config")
        missing = sorted(required - set(raw))
        if missing:
            raise ConfigError(f"config is missing keys: {', '.join(missing)}")
        training_raw = _object(raw["training"], "training")
        _unknown(training_raw, set(TRAINING_STAGES), "training")
        training = {
            name: StageConfig.from_mapping(training_raw.get(name, {}), f"training.{name}")
            for name in TRAINING_STAGES
        }
        return cls(
            project=ProjectConfig.from_mapping(raw["project"]),
            base_model=_non_empty_string(raw["base_model"], "base_model"),
            data=DataConfig.from_mapping(raw["data"]),
            training=MappingProxyType(training),
            evaluation=EvaluationConfig.from_mapping(raw["evaluation"]),
            tracking=TrackingConfig.from_mapping(raw["tracking"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": {
                "name": self.project.name,
                "domain": self.project.domain,
                "output_dir": self.project.output_dir,
                "seed": self.project.seed,
            },
            "base_model": self.base_model,
            "data": {
                "corpus_path": self.data.corpus_path,
                "general_ratio": self.data.general_ratio,
                "clean": plain_value(self.data.clean),
                "sft": plain_value(self.data.sft),
                "preference": plain_value(self.data.preference),
                "synthetic": plain_value(self.data.synthetic),
            },
            "training": {
                name: {
                    "enabled": stage.enabled,
                    "backend": stage.backend,
                    **plain_value(stage.options),
                }
                for name, stage in self.training.items()
            },
            "evaluation": {
                "enabled": self.evaluation.enabled,
                "benchmarks": list(self.evaluation.benchmarks),
                "llm_judge": plain_value(self.evaluation.llm_judge),
                "human_eval": plain_value(self.evaluation.human_eval),
            },
            "tracking": {
                "backend": self.tracking.backend,
                "project": self.tracking.project,
                "mode": self.tracking.mode,
            },
        }

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    try:
        loader = UniqueKeySafeLoader(source.read_text(encoding="utf-8"))
        try:
            raw = loader.get_single_data()
        finally:
            loader.dispose()
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration does not exist: {source}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    return ExperimentConfig.from_mapping(raw)
