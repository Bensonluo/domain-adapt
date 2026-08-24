"""Fail-closed registry for pluggable pipeline stages."""

from __future__ import annotations

import re
from collections.abc import Callable

from .types import PipelineStage

StageFactory = Callable[[], PipelineStage]
STAGE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,127}")


class StageRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, StageFactory] = {}

    def register(self, name: str, factory: StageFactory) -> None:
        if STAGE_NAME_PATTERN.fullmatch(name) is None or name in self._factories:
            raise ValueError(f"stage already registered or invalid: {name!r}")
        self._factories[name] = factory

    def create(self, name: str) -> PipelineStage:
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise KeyError(f"no implementation registered for stage {name!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
