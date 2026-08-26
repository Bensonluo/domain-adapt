#!/usr/bin/env python3
"""Validate Phase 2 Week 28 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week28/README.md",
    "week28/run_week28.sh",
    "week28/validate_week28.py",
    "week28/tests/test_week28.py",
)
COMPLETE_PATHS = (
    "adaptstack/demo/app.py",
    "adaptstack/demo/Dockerfile",
    "adaptstack/demo/nginx.conf",
    "results/week28_demo/deployment_report.md",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=28,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
