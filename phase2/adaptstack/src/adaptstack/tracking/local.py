"""Deterministic local tracking backend."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.immutable import plain_value
from ..core.io import write_json_atomic


class LocalRunTracker:
    def record(self, run_dir: Path, event: Mapping[str, Any]) -> None:
        write_json_atomic(run_dir / "tracking.json", plain_value(event))
