#!/usr/bin/env python3
"""Validate Phase 2 Week 25 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week25/README.md",
    "week25/run_week25.sh",
    "week25/validate_week25.py",
    "week25/tests/test_week25.py",
)
COMPLETE_PATHS = (
    "experiments/week25_ablation.yaml",
    "results/week25_ablation/week25_summary.json",
    "results/week25_ablation/statistics.json",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=25,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
