"""Pipeline planning and deterministic offline execution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import TRAINING_STAGES, ExperimentConfig
from .core.io import write_json_atomic
from .core.registry import StageRegistry
from .core.types import ArtifactRef, RunContext
from .stages import build_registry, relativize_artifact

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _configured_run_dir(output_dir: str, run_id: str) -> Path:
    """Resolve a config-owned output beneath the current project directory."""

    project_root = Path.cwd().resolve()
    relative = Path(output_dir)
    current = project_root
    for part in (*relative.parts, run_id):
        current /= part
        if current.is_symlink():
            raise ValueError(f"configured output path must not contain symlinks: {current}")
    resolved = current.resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError("configured output path escapes the current project directory")
    return resolved


class PipelineRunner:
    def __init__(self, config: ExperimentConfig, registry: StageRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or build_registry(config)

    def plan(self) -> tuple[str, ...]:
        stages = ["data.prepare"]
        stages.extend(
            f"training.{name}" for name in TRAINING_STAGES if self.config.training[name].enabled
        )
        if self.config.evaluation.enabled:
            stages.append("evaluation.benchmark")
        return tuple(stages)

    def run(
        self,
        *,
        output_dir: str | Path | None = None,
        dry_run: bool = False,
        run_id: str | None = None,
    ) -> Path:
        resolved_run_id = run_id or f"{self.config.project.domain}-{self.config.digest[:12]}"
        if RUN_ID_PATTERN.fullmatch(resolved_run_id) is None:
            raise ValueError(
                "run_id must be a safe identifier using letters, digits, '.', '_' or '-'"
            )
        if output_dir is None:
            run_dir = _configured_run_dir(self.config.project.output_dir, resolved_run_id)
        else:
            run_dir = Path(output_dir) / resolved_run_id
        upstream: tuple[ArtifactRef, ...] = ()
        results: list[dict[str, Any]] = []

        for index, name in enumerate(self.plan(), start=1):
            context = RunContext(
                run_id=resolved_run_id,
                domain=self.config.project.domain,
                base_model=self.config.base_model,
                config_digest=self.config.digest,
                run_dir=run_dir,
                dry_run=dry_run,
                stage_index=index,
                upstream=upstream,
            )
            result = self.registry.create(name).run(context)
            if result.stage != name:
                raise RuntimeError(
                    f"stage identity mismatch: planned {name!r}, received {result.stage!r}"
                )
            expected_status = "planned" if dry_run else "completed"
            if result.status != expected_status:
                raise RuntimeError(
                    f"stage {name!r} returned {result.status!r}; expected {expected_status!r}"
                )
            stable_outputs = tuple(relativize_artifact(item, run_dir) for item in result.outputs)
            stable_result = {
                **result.to_dict(),
                "outputs": [artifact.to_dict() for artifact in stable_outputs],
            }
            results.append(stable_result)
            upstream = stable_outputs

        summary = {
            "schema_version": 1,
            "run_id": resolved_run_id,
            "project": self.config.project.name,
            "domain": self.config.project.domain,
            "base_model": self.config.base_model,
            "config_digest": self.config.digest,
            "mode": "dry-run" if dry_run else "execution",
            "stages": list(self.plan()),
            "results": results,
        }
        write_json_atomic(run_dir / "run_summary.json", summary)
        return run_dir
