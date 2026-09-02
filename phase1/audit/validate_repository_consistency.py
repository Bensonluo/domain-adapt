#!/usr/bin/env python3
"""Validate Phase 1 JSON syntax, artifact paths and Markdown links."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE = REPO_ROOT / "phase1"
SKIP_PARTS = {".venv", ".zcode", ".ms_cache", "results", "data"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def iter_source_markdown():
    for path in PHASE.rglob("*.md"):
        if not (set(path.relative_to(PHASE).parts) & SKIP_PARTS):
            yield path


def check_markdown_links() -> int:
    checked = 0
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for path in iter_source_markdown():
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1).strip().split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            require((path.parent / target).resolve().exists(),
                    f"broken Markdown link in {path.relative_to(REPO_ROOT)}: {target}")
            checked += 1
    return checked


def check_json() -> int:
    roots = [PHASE / "audit", PHASE / "confirmation"]
    paths = [PHASE / "artifact_registry.json"]
    for root in roots:
        paths.extend(root.rglob("*.json"))
    unique = sorted(set(paths))
    for path in unique:
        json.loads(path.read_text(encoding="utf-8"))
    return len(unique)


def check_artifact_paths(value: Any, location: str = "registry") -> int:
    checked = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"summary", "lineage", "audit", "validator", "clean_reanalysis"} and isinstance(child, str):
                require((REPO_ROOT / child).exists(), f"{location}.{key} missing: {child}")
                checked += 1
            elif key == "specifications" and isinstance(child, list):
                for path in child:
                    require((REPO_ROOT / path).is_file(), f"{location}.specifications missing: {path}")
                    checked += 1
            checked += check_artifact_paths(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            checked += check_artifact_paths(child, f"{location}[{index}]")
    return checked


def main() -> None:
    registry = json.loads((PHASE / "artifact_registry.json").read_text(encoding="utf-8"))
    require(all((PHASE / f"week{week}/CORRECTIONS.md").is_file() for week in range(9, 22)),
            "one or more Week 9-21 correction logs are missing")
    json_count = check_json()
    link_count = check_markdown_links()
    artifact_count = check_artifact_paths(registry)
    print(f"PASS: Phase 1 consistency holds ({json_count} JSON, {link_count} local links, {artifact_count} artifact references)")


if __name__ == "__main__":
    main()
