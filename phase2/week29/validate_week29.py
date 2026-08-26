#!/usr/bin/env python3
"""Validate Phase 2 Week 29 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week29/README.md",
    "week29/run_week29.sh",
    "week29/validate_week29.py",
    "week29/tests/test_week29.py",
)
COMPLETE_PATHS = (
    "adaptstack/docs/quickstart.md",
    "adaptstack/docs/reproduction.md",
    "adaptstack/notebooks/cpt.ipynb",
    "adaptstack/notebooks/sft_grpo.ipynb",
    "adaptstack/notebooks/distillation.ipynb",
    "results/week29_release/release_checklist.md",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=29,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
