#!/usr/bin/env python3
"""Validate Phase 2 Week 23 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week23/README.md",
    "week23/run_week23.sh",
    "week23/validate_week23.py",
    "week23/tests/test_week23.py",
)
COMPLETE_PATHS = (
    "adaptstack/src/adaptstack/data/pipeline.py",
    "adaptstack/src/adaptstack/training/pipeline.py",
    "adaptstack/configs/medical.yaml",
    "results/week23_pipeline/week23_summary.json",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=23,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
