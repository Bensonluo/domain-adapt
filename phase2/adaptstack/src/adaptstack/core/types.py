"""Stable stage and artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .immutable import freeze_mapping, plain_value

STAGE_STATUSES = frozenset({"planned", "completed", "failed", "skipped"})


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    uri: str
    kind: str = "manifest"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "kind": self.kind,
            "metadata": plain_value(self.metadata),
        }


@dataclass(frozen=True)
class RunContext:
    run_id: str
    domain: str
    base_model: str
    config_digest: str
    run_dir: Path
    dry_run: bool
    stage_index: int
    upstream: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream", tuple(self.upstream))


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    outputs: tuple[ArtifactRef, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"unsupported stage status: {self.status!r}")
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "outputs": [artifact.to_dict() for artifact in self.outputs],
            "metrics": plain_value(self.metrics),
            "metadata": plain_value(self.metadata),
        }


class PipelineStage(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def run(self, context: RunContext) -> StageResult:
        """Execute or plan one stage and return structured artifacts."""
        ...
