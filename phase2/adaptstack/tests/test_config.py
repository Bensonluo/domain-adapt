from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from adaptstack.config import ConfigError, ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_bundled_configs_are_valid_and_stable(self) -> None:
        medical = load_config(ROOT / "configs" / "medical.yaml")
        legal = load_config(ROOT / "configs" / "legal.yaml")
        self.assertEqual(medical.project.domain, "medical")
        self.assertEqual(legal.project.domain, "legal")
        self.assertEqual(medical.digest, load_config(ROOT / "configs" / "medical.yaml").digest)
        self.assertEqual(len(medical.digest), 64)

    def test_unknown_root_key_is_rejected(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8"))
        raw["typo"] = True
        with self.assertRaisesRegex(ConfigError, "unknown keys"):
            ExperimentConfig.from_mapping(raw)

    def test_invalid_general_ratio_is_rejected(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8"))
        changed = copy.deepcopy(raw)
        changed["data"]["general_ratio"] = 1.1
        with self.assertRaisesRegex(ConfigError, "between 0 and 1"):
            ExperimentConfig.from_mapping(changed)

    def test_unsafe_domain_is_rejected(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8"))
        raw["project"]["domain"] = "../outside"
        with self.assertRaisesRegex(ConfigError, "safe identifier"):
            ExperimentConfig.from_mapping(raw)

    def test_nested_options_are_immutable(self) -> None:
        config = load_config(ROOT / "configs" / "medical.yaml")
        reward_functions = config.training["grpo"].options["reward_functions"]
        self.assertIsInstance(reward_functions, tuple)
        plain = config.to_dict()["training"]["grpo"]["reward_functions"]
        self.assertEqual(plain[0], "factuality")

    def test_unknown_training_option_is_rejected(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8"))
        raw["training"]["cpt"]["learning_rate_typo"] = 0.1
        with self.assertRaisesRegex(ConfigError, "unknown keys"):
            ExperimentConfig.from_mapping(raw)

    def test_invalid_nested_values_are_rejected(self) -> None:
        source = (ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8")
        raw = yaml.safe_load(source)
        raw["data"]["clean"]["min_length"] = -100
        with self.assertRaisesRegex(ConfigError, "positive integer"):
            ExperimentConfig.from_mapping(raw)

        raw = yaml.safe_load(source)
        raw["evaluation"]["llm_judge"]["enabled"] = "yes"
        with self.assertRaisesRegex(ConfigError, "boolean"):
            ExperimentConfig.from_mapping(raw)

    def test_backend_options_are_explicitly_extensible(self) -> None:
        raw = yaml.safe_load((ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8"))
        raw["training"]["cpt"]["backend_options"] = {"vendor_flag": "value"}
        config = ExperimentConfig.from_mapping(raw)
        self.assertEqual(
            config.to_dict()["training"]["cpt"]["backend_options"]["vendor_flag"],
            "value",
        )

    def test_configured_output_must_be_portable_and_relative(self) -> None:
        source = (ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8")
        for output_dir in ("../escaped", "/private/tmp/escaped", "."):
            raw = yaml.safe_load(source)
            raw["project"]["output_dir"] = output_dir
            with (
                self.subTest(output_dir=output_dir),
                self.assertRaisesRegex(ConfigError, "relative path"),
            ):
                ExperimentConfig.from_mapping(raw)

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        source = (ROOT / "configs" / "medical.yaml").read_text(encoding="utf-8")
        changed = source.replace("  mode: offline", "  mode: offline\n  mode: online")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8") as handle:
            handle.write(changed)
            handle.flush()
            with self.assertRaisesRegex(ConfigError, "duplicate key"):
                load_config(handle.name)


if __name__ == "__main__":
    unittest.main()
