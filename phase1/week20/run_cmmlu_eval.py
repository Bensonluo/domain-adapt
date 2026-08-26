"""Run the shared CMMLU evaluator without erasing metrics from other arms."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from artifact_lineage import verify_arms
from validate_week20 import valid_safetensors


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
WEEK20_ARMS = ("kd_t2", "kd_t5", "kd_pure", "rs_mcq", "rs_teacher", "rs_both")


def read_metric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def merged_metric(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    base_key: str,
) -> dict[str, Any]:
    """Merge prior and freshly evaluated run dictionaries; fresh values win."""
    previous_runs = previous.get("runs", {}) if previous else {}
    current_runs = current.get("runs", {})
    if not isinstance(previous_runs, dict) or not isinstance(current_runs, dict):
        raise ValueError("metric files must contain a 'runs' object")
    runs = dict(previous_runs)
    runs.update(current_runs)
    base = current.get(base_key, previous.get(base_key) if previous else None)
    return {base_key: base, "runs": runs}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fused_model_is_usable(path: Path) -> bool:
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False
    weights = [
        weight
        for pattern in ("*.safetensors", "*.bin")
        for weight in path.glob(pattern)
        if weight.is_file() and weight.stat().st_size > 0
    ]
    return bool(weights) and all(
        weight.suffix != ".safetensors" or valid_safetensors(weight) for weight in weights
    )


def metrics_cover_arms(
    sweep: Path,
    arms: list[str],
    domain: dict[str, Any] | None,
    forgetting: dict[str, Any] | None,
) -> bool:
    if not domain or not forgetting:
        return False
    domain_runs = domain.get("runs")
    forgetting_runs = forgetting.get("runs")
    if not isinstance(domain_runs, dict) or not isinstance(forgetting_runs, dict):
        return False
    for arm in arms:
        if not isinstance(domain_runs.get(arm), (int, float)) or isinstance(domain_runs.get(arm), bool):
            return False
        if not isinstance(forgetting_runs.get(arm), (int, float)) or isinstance(forgetting_runs.get(arm), bool):
            return False
        score_path = sweep / arm / f"scores_{arm}_fused.json"
        try:
            scores = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(scores, dict) or not scores:
            return False
        if not fused_model_is_usable(sweep / f"{arm}_fused"):
            return False
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preserving Week 20 CMMLU evaluator wrapper")
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--runs", nargs="+", required=True, choices=WEEK20_ARMS)
    parser.add_argument("--source-sft", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep = Path(args.sweep).resolve()
    domain_path = sweep / "domain_gain.json"
    forgetting_path = sweep / "forgetting.json"
    previous_domain = read_metric(domain_path)
    previous_forgetting = read_metric(forgetting_path)
    lineage_errors = verify_arms(
        Path(args.manifest).resolve(), sweep.resolve(), Path(args.base).resolve(),
        Path(args.source_sft).resolve(), args.runs, "evaluation"
    )
    if not lineage_errors and metrics_cover_arms(
        sweep, args.runs, previous_domain, previous_forgetting
    ):
        print(f"[cmmlu] skip: existing metrics and score artifacts cover {args.runs}")
        return 0
    for arm in args.runs:
        fused = sweep / f"{arm}_fused"
        arm_lineage_bad = any(error.startswith(f"{arm}:") for error in lineage_errors)
        if fused.exists() and (arm_lineage_bad or not fused_model_is_usable(fused)):
            quarantine = sweep / f".{fused.name}.incomplete-{os.getpid()}"
            os.replace(fused, quarantine)
            print(f"[cmmlu] moved incomplete fused artifact aside: {fused} -> {quarantine}")
    base_score = sweep / "base_hf" / f"scores_{Path(args.base).name}.json"
    if lineage_errors and base_score.exists():
        quarantine = base_score.with_name(f".{base_score.name}.stale-{os.getpid()}")
        os.replace(base_score, quarantine)
        print(f"[cmmlu] moved stale base scores aside: {base_score} -> {quarantine}")
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "phase1/week15/run_dpo_eval.py"),
        "--sweep",
        str(sweep),
        "--base",
        args.base,
        "--runs",
        *args.runs,
    ]
    if not lineage_errors and base_score.is_file():
        command.append("--skip-base")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        # The evaluator normally writes only after all arms succeed. Restore a
        # prior aggregate defensively if a future version writes incrementally.
        if previous_domain is not None:
            atomic_write_json(domain_path, previous_domain)
        if previous_forgetting is not None:
            atomic_write_json(forgetting_path, previous_forgetting)
        return completed.returncode

    current_domain = read_metric(domain_path)
    current_forgetting = read_metric(forgetting_path)
    if current_domain is None or current_forgetting is None:
        raise RuntimeError("CMMLU evaluator completed without metric files")
    atomic_write_json(
        domain_path,
        merged_metric(None if lineage_errors else previous_domain, current_domain, "base_medical_cn"),
    )
    atomic_write_json(
        forgetting_path,
        merged_metric(
            None if lineage_errors else previous_forgetting,
            current_forgetting,
            "base_general_cn",
        ),
    )
    print(f"[cmmlu] preserved aggregate metrics for {len(args.runs)} freshly evaluated arm(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
