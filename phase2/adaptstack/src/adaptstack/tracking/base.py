"""Experiment tracking protocol."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class RunTracker(Protocol):
    def record(self, run_dir: Path, event: Mapping[str, Any]) -> None:
        """Persist one structured tracking event."""
        ...
