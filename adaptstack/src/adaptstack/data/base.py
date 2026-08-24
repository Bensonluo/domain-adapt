"""Backend-neutral data contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.immutable import freeze_mapping


@dataclass(frozen=True)
class DataRecord:
    record_id: str
    text: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class DataSource(Protocol):
    def records(self) -> Iterable[DataRecord]:
        """Yield normalized records without assuming a storage backend."""
        ...
