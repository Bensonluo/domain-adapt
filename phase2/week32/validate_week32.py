#!/usr/bin/env python3
"""Validate Phase 2 Week 32 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week32/README.md",
    "week32/run_week32.sh",
    "week32/validate_week32.py",
    "week32/tests/test_week32.py",
)
COMPLETE_PATHS = (
    "results/week32_retrospective/deliverables.md",
    "results/week32_retrospective/domain_boundary.md",
    "results/week32_retrospective/talk_outline.md",
    "results/week32_retrospective/phase2_summary.md",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=32,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
