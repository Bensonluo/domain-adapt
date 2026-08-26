#!/usr/bin/env python3
"""Validate Phase 2 Week 27 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week27/README.md",
    "week27/run_week27.sh",
    "week27/validate_week27.py",
    "week27/tests/test_week27.py",
)
COMPLETE_PATHS = (
    "results/week27_insights/insights.md",
    "results/week27_insights/evidence_index.json",
    "results/week27_insights/limitations.md",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=27,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
