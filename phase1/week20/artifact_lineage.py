"""Content-hash lineage manifests for Week 20 checkpoints and evaluations."""

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
REPO_ROOT = HERE.parent.parent
ARMS = ("kd_t2", "kd_t5", "kd_pure", "rs_mcq", "rs_teacher", "rs_both")
KD_ARMS = ARMS[:3]


@lru_cache(maxsize=256)
def _digest_file(path_string: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # cache-key provenance; content is read from path_string
    digest = hashlib.sha256()
    with Path(path_string).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_files(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in files}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_digest_file(str(path), stat.st_size, stat.st_mtime_ns).encode())
    return digest.hexdigest()


def model_files(directory: Path) -> list[Path]:
    metadata = [
        directory / name
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        if (directory / name).is_file()
    ]
    weights = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in directory.glob(pattern)
        if path.is_file()
    ]
    if not weights:
        raise FileNotFoundError(f"no model weights in {directory}")
    return metadata + weights


def current_lineage(sweep: Path, base: Path, source_sft: Path, arm: str, stage: str) -> dict[str, Any]:
    config_path = sweep / arm / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    dependency_files = model_files(base) + [data_path]
    if arm in KD_ARMS:
        logits_path = Path(config["logits"])
        if not logits_path.is_absolute():
            logits_path = REPO_ROOT / logits_path
        dependency_files.append(logits_path)
    else:
        dependency_files.extend(
            [sweep / "data/student_samples.jsonl", sweep / "data/judge_scores.jsonl"]
        )
    checkpoint = {
        "inputs_sha256": digest_files(dependency_files),
        "run_config_sha256": digest_files([config_path]),
        "adapter_sha256": digest_files([sweep / arm / "adapter_model.safetensors"]),
    }
    record: dict[str, Any] = {"checkpoint": checkpoint}
    if stage == "evaluation":
        fused = sweep / f"{arm}_fused"
        evaluation_files = [
            fused / "cmexam_holdout.json",
            fused / "cmexam_holdout.json.preds.jsonl",
            sweep / arm / f"scores_{arm}_fused.json",
            sweep / "base_hf" / f"scores_{base.name}.json",
        ]
        record["evaluation"] = {
            "checkpoint_sha256": hashlib.sha256(
                json.dumps(checkpoint, sort_keys=True).encode()
            ).hexdigest(),
            "fused_sha256": digest_files(model_files(fused)),
            "evaluation_sha256": digest_files(evaluation_files),
        }
    return record


def current_data_lineage(
    sweep: Path, base: Path, source_sft: Path, teacher: Path, stage: str
) -> dict[str, Any]:
    if stage == "samples":
        inputs = model_files(base) + [source_sft, HERE / "generate_student_samples.py"]
        output = [sweep / "data/student_samples.jsonl"]
        parameters = {"n": 8, "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 128, "seed": 123}
    elif stage == "scores":
        inputs = model_files(teacher) + [
            sweep / "data/student_samples.jsonl", HERE / "judge_with_teacher.py"
        ]
        output = [sweep / "data/judge_scores.jsonl"]
        parameters = {"max_tokens": 96, "temperature": 0.0}
    elif stage == "logits":
        inputs = model_files(teacher) + [source_sft, HERE / "extract_teacher_logits.py"]
        output = [sweep / "data/teacher_topk_logits.jsonl"]
        parameters = {"topk": 20}
    else:
        raise ValueError(f"unknown data stage: {stage}")
    return {
        "inputs_sha256": digest_files(inputs),
        "parameters_sha256": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
        "output_sha256": digest_files(output),
    }


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "arms": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != 1 or not isinstance(value.get("arms"), dict):
        raise ValueError(f"invalid lineage manifest: {path}")
    return value


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_arms(manifest_path: Path, sweep: Path, base: Path, source_sft: Path,
                arms: list[str], stage: str) -> None:
    manifest = read_manifest(manifest_path)
    for arm in arms:
        manifest["arms"][arm] = current_lineage(sweep, base, source_sft, arm, stage)
    write_manifest(manifest_path, manifest)


def verify_arms(manifest_path: Path, sweep: Path, base: Path, source_sft: Path,
                arms: list[str], stage: str) -> list[str]:
    try:
        manifest = read_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]
    errors = []
    for arm in arms:
        expected = manifest["arms"].get(arm, {}).get(stage)
        try:
            actual = current_lineage(sweep, base, source_sft, arm, stage).get(stage)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{arm}: cannot compute {stage} lineage: {exc}")
            continue
        if expected != actual:
            errors.append(f"{arm}: {stage} content lineage mismatch")
    return errors


def record_data_stage(manifest_path: Path, sweep: Path, base: Path, source_sft: Path,
                      teacher: Path, stage: str) -> None:
    manifest = read_manifest(manifest_path)
    manifest.setdefault("data", {})[stage] = current_data_lineage(
        sweep, base, source_sft, teacher, stage
    )
    write_manifest(manifest_path, manifest)


def verify_data_stage(manifest_path: Path, sweep: Path, base: Path, source_sft: Path,
                      teacher: Path, stage: str, inputs_only: bool = False) -> list[str]:
    try:
        manifest = read_manifest(manifest_path)
        expected = manifest.get("data", {}).get(stage)
        actual = current_data_lineage(sweep, base, source_sft, teacher, stage)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return [f"{stage}: cannot compute producer lineage: {exc}"]
    if inputs_only and isinstance(expected, dict):
        expected = {key: value for key, value in expected.items() if key != "output_sha256"}
        actual = {key: value for key, value in actual.items() if key != "output_sha256"}
    return [] if expected == actual else [f"{stage}: producer content lineage mismatch"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record or verify Week 20 content lineage")
    parser.add_argument("action", choices=("record", "verify"))
    parser.add_argument("--manifest", default=str(HERE / "week20_lineage.json"))
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--source-sft", required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--stage", choices=("logits", "samples", "scores", "checkpoint", "evaluation"), required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    paths = (
        Path(args.manifest).resolve(), Path(args.sweep).resolve(),
        Path(args.base).resolve(), Path(args.source_sft).resolve(),
    )
    if args.stage in ("logits", "samples", "scores"):
        if not args.teacher:
            parser.error(f"--teacher is required for {args.stage} lineage")
        teacher = Path(args.teacher).resolve()
        if args.action == "record":
            record_data_stage(*paths, teacher, args.stage)
            print(f"[lineage] recorded producer stage: {args.stage}")
            return 0
        errors = verify_data_stage(*paths, teacher, args.stage, args.inputs_only)
    else:
        if not args.arms:
            parser.error(f"--arms is required for {args.stage} lineage")
        if args.action == "record":
            record_arms(*paths, args.arms, args.stage)
            print(f"[lineage] recorded {args.stage}: {args.arms}")
            return 0
        errors = verify_arms(*paths, args.arms, args.stage)
    for error in errors:
        print(f"[lineage:error] {error}")
    if not errors:
        print(f"[lineage] verified {args.stage}: {args.arms or 'producer'}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
