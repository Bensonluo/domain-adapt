from __future__ import annotations

import unittest
from pathlib import Path

from adaptstack.config import load_config
from adaptstack.core.types import ArtifactRef, RunContext, StageResult
from adaptstack.stages import build_registry

ROOT = Path(__file__).resolve().parents[1]


class ImmutabilityTests(unittest.TestCase):
    def test_stage_settings_cannot_diverge_from_config_digest(self) -> None:
        config = load_config(ROOT / "configs" / "medical.yaml")
        stage = build_registry(config).create("data.prepare")
        with self.assertRaises(TypeError):
            stage.settings["general_ratio"] = 0.99  # type: ignore[index,union-attr]

    def test_artifact_metadata_is_deeply_immutable(self) -> None:
        artifact = ArtifactRef("input", "input.json", metadata={"nested": {"values": [1]}})
        nested = artifact.metadata["nested"]
        with self.assertRaises(TypeError):
            nested["changed"] = True  # type: ignore[index]
        with self.assertRaises(AttributeError):
            nested["values"].append(2)  # type: ignore[attr-defined,index]

    def test_result_and_context_copy_mutable_containers(self) -> None:
        artifact = ArtifactRef("input", "input.json")
        upstream = [artifact]
        context = RunContext("run", "medical", "model", "digest", ROOT, True, 1, upstream)
        upstream.clear()
        self.assertEqual(context.upstream, (artifact,))

        metrics = {"accuracy": 0.5}
        result = StageResult("eval", "completed", (), metrics=metrics)
        metrics["accuracy"] = 0.9
        self.assertEqual(result.metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
