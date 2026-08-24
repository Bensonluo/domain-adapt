"""Backend-neutral inference contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.immutable import freeze_mapping


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    domain: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True)
class InferenceResponse:
    text: str
    model: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class InferenceBackend(Protocol):
    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate one response using a configured serving backend."""
        ...
