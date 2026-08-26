"""Backend-neutral evaluation contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..core.immutable import freeze_mapping


@dataclass(frozen=True)
class EvaluationResult:
    benchmark: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", freeze_mapping(self.metrics))


class Evaluator(Protocol):
    def evaluate(self, model_uri: str, benchmarks: Sequence[str]) -> Sequence[EvaluationResult]:
        """Evaluate one model artifact against named benchmarks."""
        ...
