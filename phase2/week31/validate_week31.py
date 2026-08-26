#!/usr/bin/env python3
"""Validate Phase 2 Week 31 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week31/README.md",
    "week31/run_week31.sh",
    "week31/validate_week31.py",
    "week31/tests/test_week31.py",
)
COMPLETE_PATHS = (
    "results/week31_community/release_links.md",
    "results/week31_community/model_card.md",
    "results/week31_community/dataset_card.md",
    "results/week31_community/feedback_summary.md",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=31,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
