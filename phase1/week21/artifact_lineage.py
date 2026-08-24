"""Content-hash lineage for every material Week 21 experiment stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in HERE.parents if (parent / "phase1").is_dir())
STAGES = ("seeds", "self", "evol", "quality", "replacement", "checkpoint", "evaluation", "summary")


@lru_cache(maxsize=512)
def digest_file(path_string: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    stat = path.stat()
    return digest_file(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def model_files(directory: Path) -> list[Path]:
    files = [
        path for path in directory.iterdir()
        if path.is_file() and (
            path.suffix in (".safetensors", ".bin", ".json", ".jinja")
            or path.name.endswith(".model")
        )
    ]
    if not any(path.suffix in (".safetensors", ".bin") for path in files):
        raise FileNotFoundError(f"no model weights in {directory}")
    return files


def digest_named(items: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(items, key=lambda item: item[0]):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def named_model(prefix: str, directory: Path) -> list[tuple[str, Path]]:
    return [(f"{prefix}/{path.name}", path) for path in model_files(directory)]


def jsonl_all_have(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    found = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            found = True
            if key not in json.loads(line):
                return False
    return found


def jsonl_values(path: Path, key: str) -> list[str]:
    if not path.is_file():
        return []
    values = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line).get(key)
                if value:
                    values.add(str(value))
    return sorted(values)


def stage_material(
    stage: str,
    sweep: Path,
    base: Path,
    teacher: Path,
    control_model: Path,
    n_self: int,
    n_evol: int,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]], dict[str, Any]]:
    data = sweep / "data"
    if stage == "seeds":
        inputs = [
            ("train", REPO_ROOT / "phase1/data/processed/cmexam/train.jsonl"),
            ("holdout", REPO_ROOT / "phase1/data/processed/cmexam/test.jsonl"),
            ("common_code", HERE / "common.py"), ("producer_code", HERE / "prepare_seeds.py"),
        ]
        outputs = [("seeds", data / "seed_instructions.jsonl")]
        params = {"n": 100, "seed": 123, "holdout_filter": "exact_question"}
    elif stage == "self":
        inputs = named_model("teacher", teacher) + [
            ("seeds", data / "seed_instructions.jsonl"),
            ("common_code", HERE / "common.py"), ("producer_code", HERE / "self_instruct.py"),
        ]
        outputs = [("raw", data / "self_instruct_raw.jsonl"), ("accepted", data / "self_instruct.jsonl"), ("stats", data / "self_instruct_stats.json")]
        params = {
            "backend": "mlx", "n": n_self, "batch_size": 4, "similarity_threshold": 0.78,
            "temperature": 0.2, "seed": 123,
            "raw_mlx_rng_seeded": jsonl_all_have(data / "self_instruct_raw.jsonl", "rng_seed"),
        }
    elif stage == "evol":
        inputs = named_model("teacher", teacher) + [
            ("self_accepted", data / "self_instruct.jsonl"),
            ("common_code", HERE / "common.py"), ("producer_code", HERE / "evol_instruct.py"),
        ]
        outputs = [("raw", data / "evol_instruct_raw.jsonl"), ("accepted", data / "evol_instruct.jsonl"), ("stats", data / "evol_instruct_stats.json")]
        params = {
            "backend": "mlx", "n": n_evol, "max_depth": 3, "similarity_threshold": 0.98,
            "temperature": 0.1, "seed": 123,
            "raw_mlx_rng_seeded": jsonl_all_have(data / "evol_instruct_raw.jsonl", "rng_seed"),
        }
    elif stage == "quality":
        inputs = [
            ("self", data / "self_instruct.jsonl"), ("evol", data / "evol_instruct.jsonl"),
            ("real", REPO_ROOT / "phase1/data/processed/cmexam/train.jsonl"),
            ("quality_code", HERE / "assess_quality.py"),
            ("manual_review_record", HERE / "apply_audit_labels.py"),
        ]
        outputs = [("metrics", sweep / "quality_metrics.json"), ("audit", sweep / "human_audit.jsonl")]
        params = {
            "comparison_sample": 5000, "audit_size": 30, "seed": 123,
            "reviewer_kinds": jsonl_values(sweep / "human_audit.jsonl", "reviewer_kind"),
        }
    elif stage == "replacement":
        inputs = [
            ("real_sft", REPO_ROOT / "phase1/results/week19_distill/data/real_sft.jsonl"),
            ("self", data / "self_instruct.jsonl"), ("evol", data / "evol_instruct.jsonl"),
            ("holdout", REPO_ROOT / "phase1/data/processed/cmexam/test.jsonl"),
            ("producer_code", HERE / "prepare_replacement.py"),
            ("similarity_code", HERE / "check_holdout_similarity.py"),
        ]
        outputs = [
            ("treatment", data / "replacement_50.jsonl"),
            ("manifest", data / "replacement_50.manifest.json"),
            ("holdout_similarity", sweep / "holdout_similarity.json"),
        ]
        params = {"n_total": 2000, "synthetic_fraction": 0.5, "seed": 123}
    elif stage == "checkpoint":
        inputs = named_model("base", base) + [
            ("treatment", data / "replacement_50.jsonl"),
            ("trainer_code", REPO_ROOT / "phase1/week19/train_distill_sft.py"),
        ]
        outputs = named_model("adapter", sweep / "synthetic50") + [
            ("run_config", sweep / "synthetic50/run_config.json"),
            ("loss_log", sweep / "synthetic50/loss_log.csv"),
        ]
        params = {"lr": 2e-5, "epochs": 3.0, "batch_size": 4, "grad_accum": 4, "max_length": 1536, "seed": 123}
    elif stage == "evaluation":
        inputs = named_model("base", base) + named_model("control_model", control_model) + named_model("adapter", sweep / "synthetic50") + [
            ("control_run_config", REPO_ROOT / "phase1/results/week19_distill/real/run_config.json"),
            ("control_training_data", REPO_ROOT / "phase1/results/week19_distill/data/real_sft.jsonl"),
            ("merge_code", HERE / "merge_adapter.py"),
            ("eval_code", HERE / "eval_cmexam_full.py"),
            ("statistics_code", HERE / "analyze_replacement.py"),
            ("holdout", REPO_ROOT / "phase1/data/processed/cmexam/holdout.jsonl"),
        ]
        outputs = named_model("fused", sweep / "synthetic50_fused") + [
            ("evaluation", sweep / "synthetic50_fused/cmexam_holdout.json"),
            ("predictions", sweep / "synthetic50_fused/cmexam_holdout.json.preds.jsonl"),
            ("control_evaluation", sweep / "control_real_fused/cmexam_holdout.json"),
            ("control_predictions", sweep / "control_real_fused/cmexam_holdout.json.preds.jsonl"),
            ("statistics", sweep / "replacement_statistics.json"),
        ]
        params = {
            "holdout_n": 500, "max_new_tokens": 16, "batch_size": 8,
            "device": "cpu", "dtype": "float32", "bootstrap_samples": 20000, "margin": -0.02,
        }
    elif stage == "summary":
        inputs = [
            ("quality", sweep / "quality_metrics.json"), ("audit", sweep / "human_audit.jsonl"),
            ("holdout_similarity", sweep / "holdout_similarity.json"),
            ("replacement_manifest", data / "replacement_50.manifest.json"),
            ("treatment_eval", sweep / "synthetic50_fused/cmexam_holdout.json"),
            ("control_eval", sweep / "control_real_fused/cmexam_holdout.json"),
            ("statistics", sweep / "replacement_statistics.json"),
            ("summary_code", HERE / "summarize_week21.py"),
        ]
        outputs = [
            ("summary", sweep / "week21_summary.json"),
            ("quality_report", HERE / "results/week21_synthetic_quality.md"),
            ("replacement_report", HERE / "results/week21_replacement_experiment.md"),
        ]
        params = {"acceptance_threshold": -0.02, "control": "week19_real_2000"}
    else:
        raise ValueError(stage)
    return inputs, outputs, params


def current_stage(stage: str, sweep: Path, base: Path, teacher: Path, control_model: Path,
                  n_self: int, n_evol: int, inputs_only: bool = False) -> dict[str, Any]:
    inputs, outputs, params = stage_material(stage, sweep, base, teacher, control_model, n_self, n_evol)
    record = {
        "inputs_sha256": digest_named(inputs),
        "parameters_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
        "parameters": params,
    }
    if not inputs_only:
        record["outputs_sha256"] = digest_named(outputs)
    return record


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "stages": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1 or not isinstance(value.get("stages"), dict):
        raise ValueError(f"invalid lineage manifest: {path}")
    return value


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--control-model", type=Path, required=True)
    parser.add_argument("--n-self", type=int, default=1000)
    parser.add_argument("--n-evol", type=int, default=250)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    paths = [args.manifest, args.sweep, args.base, args.teacher, args.control_model]
    manifest_path, sweep, base, teacher, control_model = [path.resolve() for path in paths]
    try:
        actual = current_stage(args.stage, sweep, base, teacher, control_model, args.n_self, args.n_evol, args.inputs_only)
        manifest = read_manifest(manifest_path)
        if args.action == "record":
            manifest["stages"][args.stage] = actual
            write_manifest(manifest_path, manifest)
            print(f"[lineage] recorded {args.stage}{' inputs' if args.inputs_only else ''}")
            return 0
        expected = manifest["stages"].get(args.stage)
        if args.inputs_only and isinstance(expected, dict):
            expected = {key: expected.get(key) for key in ("inputs_sha256", "parameters_sha256", "parameters")}
        if expected != actual:
            print(f"[lineage:error] {args.stage} content lineage mismatch")
            return 1
        print(f"[lineage] verified {args.stage}{' inputs' if args.inputs_only else ''}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[lineage:error] {args.stage}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
