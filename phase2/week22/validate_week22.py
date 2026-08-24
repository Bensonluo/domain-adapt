#!/usr/bin/env python3
"""Validate Phase 2 Week 22 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week22/README.md",
    "week22/run_week22.sh",
    "week22/validate_week22.py",
    "week22/tests/test_week22.py",
    "adaptstack/pyproject.toml",
    "adaptstack/src/adaptstack/pipeline.py",
    "adaptstack/tests/test_pipeline.py",
)
COMPLETE_PATHS = (
    "adaptstack/docs/architecture.md",
    "adaptstack/docs/module-interfaces.md",
    "adaptstack/README.md",
    "adaptstack/CONTRIBUTING.md",
    "adaptstack/scripts/train.py",
    "adaptstack/tests/test_pipeline.py",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=22,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
