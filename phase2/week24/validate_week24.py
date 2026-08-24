#!/usr/bin/env python3
"""Validate Phase 2 Week 24 without rerunning expensive work."""

from __future__ import annotations

import sys
from pathlib import Path

PHASE2_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from utils.week_validation import validate_week  # noqa: E402

CODE_PATHS = (
    "week24/README.md",
    "week24/run_week24.sh",
    "week24/validate_week24.py",
    "week24/tests/test_week24.py",
)
COMPLETE_PATHS = (
    "adaptstack/src/adaptstack/inference/vllm_deploy/runner.py",
    "adaptstack/src/adaptstack/inference/rag/pipeline.py",
    "adaptstack/src/adaptstack/eval/pipeline.py",
    "results/week24_first_run/week24_summary.json",
)


if __name__ == "__main__":
    raise SystemExit(
        validate_week(
            week_file=__file__,
            week=24,
            code_paths=CODE_PATHS,
            complete_paths=COMPLETE_PATHS,
        )
    )
