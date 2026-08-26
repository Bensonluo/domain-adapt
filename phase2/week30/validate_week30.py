#!/usr/bin/env python3
"""Validate Phase 2 Week 30 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week30/README.md",
    "week30/run_week30.sh",
    "week30/validate_week30.py",
    "week30/tests/test_week30.py",
)
COMPLETE_PATHS = (
    "paper/main.tex",
    "paper/references.bib",
    "paper/figures",
    "results/week30_paper/build_report.md",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=30,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
