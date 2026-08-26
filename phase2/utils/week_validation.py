#!/usr/bin/env python3
"""Shared, fail-closed validation helpers for Phase 2 weeks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


class ValidationError(ValueError):
    """Raised when a weekly deliverable is missing or unsafe."""


def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError(f"unsafe project-relative path: {relative}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValidationError(f"path escapes Phase 2 root: {relative}")
    return resolved


def _check_paths(root: Path, paths: Sequence[str]) -> list[str]:
    missing: list[str] = []
    for relative in paths:
        path = _safe_path(root, relative)
        if not path.exists():
            missing.append(relative)
        elif path.is_file() and path.stat().st_size == 0:
            missing.append(f"{relative} (empty)")
        elif path.is_dir() and not any(path.iterdir()):
            missing.append(f"{relative} (empty directory)")
    return missing


def validate_week(
    *,
    week_file: str,
    week: int,
    code_paths: Sequence[str],
    complete_paths: Sequence[str],
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=f"Validate Phase 2 Week {week}")
    parser.add_argument("--scope", choices=("code", "complete"), default="code")
    args = parser.parse_args(argv)

    root = Path(week_file).resolve().parents[1]
    required = list(code_paths)
    if args.scope == "complete":
        required.extend(complete_paths)

    try:
        missing = _check_paths(root, required)
    except (OSError, ValidationError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if missing:
        print(f"FAILED: Week {week} {args.scope} validation")
        for relative in missing:
            print(f"  - missing: {relative}")
        return 1

    print(f"VALIDATED: Week {week} {args.scope}")
    return 0
