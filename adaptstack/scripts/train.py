#!/usr/bin/env python3
"""Compatibility entry point for Week 22 examples and automation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adaptstack.cli import doctor  # noqa: E402
from adaptstack.config import ConfigError, load_config  # noqa: E402
from adaptstack.pipeline import PipelineRunner  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or dry-run an AdaptStack experiment")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-plan", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.doctor:
        return doctor(PROJECT_ROOT)
    if args.config is None:
        print("error: --config is required unless --doctor is used", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        runner = PipelineRunner(config)
        if args.show_plan:
            print(json.dumps({"config_digest": config.digest, "stages": runner.plan()}, indent=2))
            return 0
        run_dir = runner.run(output_dir=args.output, dry_run=args.dry_run)
        print(json.dumps({"mode": "dry-run", "run_dir": str(run_dir)}, indent=2))
        return 0
    except (ConfigError, OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
