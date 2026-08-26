#!/usr/bin/env python3
"""Validate Phase 2 Week 26 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week26/README.md",
    "week26/run_week26.sh",
    "week26/validate_week26.py",
    "week26/tests/test_week26.py",
)
COMPLETE_PATHS = (
    "experiments/week26_ablation.yaml",
    "configs/legal.yaml",
    "results/week26_cross_domain/week26_summary.json",
    "results/week26_cross_domain/statistics.json",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=26,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
