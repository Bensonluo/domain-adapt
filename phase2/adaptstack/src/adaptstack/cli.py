"""Command-line interface for validating and planning AdaptStack runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from .config import ConfigError, load_config
from .pipeline import PipelineRunner


class DoctorCheck(TypedDict):
    ok: bool
    detail: str


def doctor(project_root: Path) -> int:
    checks: dict[str, DoctorCheck] = {
        "python": {
            "ok": sys.version_info >= (3, 10),
            "detail": platform.python_version(),
        },
        "pyyaml": {
            "ok": importlib.util.find_spec("yaml") is not None,
            "detail": "installed" if importlib.util.find_spec("yaml") else "missing",
        },
    }
    for name in ("medical.yaml", "legal.yaml"):
        path = project_root / "configs" / name
        try:
            load_config(path)
            checks[f"config:{name}"] = {"ok": True, "detail": str(path)}
        except (ConfigError, OSError) as exc:
            checks[f"config:{name}"] = {"ok": False, "detail": str(exc)}
    print(json.dumps(checks, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item["ok"] for item in checks.values()) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptstack", description="AdaptStack scaffold CLI")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="check the local scaffold")
    plan = subparsers.add_parser("plan", help="print an ordered stage plan")
    plan.add_argument("--config", required=True, type=Path)
    run = subparsers.add_parser("run", help="materialize an offline run plan")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args.project_root.resolve())
        config = load_config(args.config)
        runner = PipelineRunner(config)
        if args.command == "plan":
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
