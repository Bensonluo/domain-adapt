from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from adaptstack.config import load_config
from adaptstack.core.registry import StageRegistry
from adaptstack.core.types import RunContext, StageResult
from adaptstack.pipeline import PipelineRunner

ROOT = Path(__file__).resolve().parents[1]


class ExecutableStage:
    def __init__(
        self, name: str, *, reported_name: str | None = None, status: str = "completed"
    ) -> None:
        self.name = name
        self.kind = "test"
        self.reported_name = reported_name or name
        self.status = status

    def run(self, context: RunContext) -> StageResult:
        self.last_dry_run = context.dry_run
        return StageResult(stage=self.reported_name, status=self.status, outputs=())


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "medical.yaml")
        self.runner = PipelineRunner(self.config)

    def test_plan_contains_only_enabled_stages_in_order(self) -> None:
        self.assertEqual(
            self.runner.plan(),
            (
                "data.prepare",
                "training.cpt",
                "training.sft",
                "training.grpo",
                "evaluation.benchmark",
            ),
        )

    def test_dry_run_is_deterministic_and_tracks_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.runner.run(output_dir=temp_dir, dry_run=True)
            before = (run_dir / "run_summary.json").read_bytes()
            second_run_dir = self.runner.run(output_dir=temp_dir, dry_run=True)
            after = (second_run_dir / "run_summary.json").read_bytes()
            self.assertEqual(before, after)
            manifests = sorted((run_dir / "manifests").glob("*.json"))
            self.assertEqual(len(manifests), len(self.runner.plan()))
            second_manifest = json.loads(manifests[1].read_text(encoding="utf-8"))
            self.assertEqual(second_manifest["upstream"][0]["name"], "data.prepare.manifest")
            self.assertEqual(second_manifest["config_digest"], self.config.digest)

    def test_execution_without_dry_run_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--dry-run"):
            self.runner.run()

    def test_custom_registry_can_execute_without_dry_run(self) -> None:
        registry = StageRegistry()
        for name in self.runner.plan():
            registry.register(name, lambda stage_name=name: ExecutableStage(stage_name))
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = PipelineRunner(self.config, registry).run(output_dir=temp_dir)
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "execution")
            self.assertTrue(all(item["status"] == "completed" for item in summary["results"]))

    def test_execution_rejects_failed_stage(self) -> None:
        registry = StageRegistry()
        for name in self.runner.plan():
            status = "failed" if name == "training.sft" else "completed"
            registry.register(
                name,
                lambda stage_name=name, result_status=status: ExecutableStage(
                    stage_name, status=result_status
                ),
            )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(RuntimeError, "returned 'failed'"),
        ):
            PipelineRunner(self.config, registry).run(output_dir=temp_dir)

    def test_execution_rejects_stage_identity_mismatch(self) -> None:
        registry = StageRegistry()
        for name in self.runner.plan():
            reported_name = "wrong.stage" if name == "training.sft" else name
            registry.register(
                name,
                lambda stage_name=name, result_name=reported_name: ExecutableStage(
                    stage_name, reported_name=result_name
                ),
            )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(RuntimeError, "identity mismatch"),
        ):
            PipelineRunner(self.config, registry).run(output_dir=temp_dir)

    def test_path_like_run_id_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe identifier"):
            self.runner.run(dry_run=True, run_id="../outside")

    def test_configured_output_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "artifacts").symlink_to(root.parent, target_is_directory=True)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with self.assertRaisesRegex(ValueError, "symlinks"):
                    self.runner.run(dry_run=True)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
