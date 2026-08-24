"""Built-in dry-run stages used by the Week 22 scaffold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .core.immutable import freeze_mapping, plain_value
from .core.io import write_json_atomic
from .core.registry import StageFactory, StageRegistry
from .core.types import ArtifactRef, PipelineStage, RunContext, StageResult


def _slug(value: str) -> str:
    return value.replace(".", "-").replace("_", "-")


@dataclass(frozen=True)
class ManifestStage:
    """Materialize a deterministic stage plan without invoking a backend."""

    name: str
    kind: str
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", freeze_mapping(self.settings))

    def run(self, context: RunContext) -> StageResult:
        if not context.dry_run:
            raise RuntimeError(
                f"stage {self.name!r} has no execution backend; rerun with --dry-run "
                "or register a production implementation"
            )

        manifest_path = (
            context.run_dir / "manifests" / f"{context.stage_index:02d}-{_slug(self.name)}.json"
        )
        manifest = {
            "schema_version": 1,
            "stage": self.name,
            "kind": self.kind,
            "status": "planned",
            "run_id": context.run_id,
            "domain": context.domain,
            "base_model": context.base_model,
            "config_digest": context.config_digest,
            "dry_run": True,
            "upstream": [artifact.to_dict() for artifact in context.upstream],
            "settings": plain_value(self.settings),
        }
        write_json_atomic(manifest_path, manifest)
        artifact = ArtifactRef(
            name=f"{self.name}.manifest",
            uri=str(manifest_path),
            metadata={"stage": self.name, "config_digest": context.config_digest},
        )
        return StageResult(stage=self.name, status="planned", outputs=(artifact,))


def _manifest_factory(name: str, kind: str, settings: Mapping[str, Any]) -> StageFactory:
    def create() -> PipelineStage:
        return ManifestStage(name, kind, settings)

    return create


def build_registry(config: ExperimentConfig) -> StageRegistry:
    """Create the default registry for one immutable experiment config."""

    registry = StageRegistry()
    data_settings = {
        "corpus_path": config.data.corpus_path,
        "general_ratio": config.data.general_ratio,
        "clean": plain_value(config.data.clean),
        "sft": plain_value(config.data.sft),
        "preference": plain_value(config.data.preference),
        "synthetic": plain_value(config.data.synthetic),
    }
    registry.register("data.prepare", _manifest_factory("data.prepare", "data", data_settings))
    for stage_name, stage_config in config.training.items():
        settings = {"backend": stage_config.backend, **plain_value(stage_config.options)}
        name = f"training.{stage_name}"
        registry.register(name, _manifest_factory(name, "training", settings))
    evaluation_settings = {
        "benchmarks": list(config.evaluation.benchmarks),
        "llm_judge": plain_value(config.evaluation.llm_judge),
        "human_eval": plain_value(config.evaluation.human_eval),
    }
    registry.register(
        "evaluation.benchmark",
        _manifest_factory("evaluation.benchmark", "evaluation", evaluation_settings),
    )
    return registry


def relativize_artifact(artifact: ArtifactRef, run_dir: Path) -> ArtifactRef:
    """Return a stable, run-relative artifact reference for summaries."""

    path = Path(artifact.uri)
    try:
        uri = str(path.relative_to(run_dir))
    except ValueError:
        uri = artifact.uri
    return ArtifactRef(artifact.name, uri, artifact.kind, artifact.metadata)
