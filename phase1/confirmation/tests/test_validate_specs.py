from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


CONFIRMATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONFIRMATION))

from validate_specs import SPEC_FILES, SpecError, load_and_validate_all, validate_spec  # noqa: E402


class ConfirmationSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {
            filename: json.loads((CONFIRMATION / filename).read_text(encoding="utf-8"))
            for filename in SPEC_FILES
        }

    def test_all_frozen_specs_validate(self):
        self.assertEqual(len(load_and_validate_all()), 5)

    def test_rejects_single_seed(self):
        spec = copy.deepcopy(self.specs["cpt.json"])
        spec["seeds"] = [101]
        with self.assertRaisesRegex(SpecError, "3 unique seeds"):
            validate_spec(spec)

    def test_rejects_mutable_base_revision(self):
        spec = copy.deepcopy(self.specs["cpt.json"])
        spec["base_model"]["revision"] = "main"
        with self.assertRaisesRegex(SpecError, "immutable commit"):
            validate_spec(spec)

    def test_rejects_incomplete_training_recipe(self):
        spec = copy.deepcopy(self.specs["grpo.json"])
        spec["training_recipe"] = {"optimizer": "adamw"}
        with self.assertRaisesRegex(SpecError, "not executable"):
            validate_spec(spec)

    def test_rejects_historical_holdout_as_confirmation(self):
        spec = copy.deepcopy(self.specs["grpo.json"])
        spec["data"]["confirmation"]["path"] = "phase1/data/processed/cmexam/holdout.jsonl"
        with self.assertRaisesRegex(SpecError, "historical holdout"):
            validate_spec(spec)

    def test_rejects_historical_holdout_path_alias(self):
        spec = copy.deepcopy(self.specs["grpo.json"])
        spec["data"]["confirmation"]["path"] = "phase1/data/processed/cmexam/../cmexam/holdout.jsonl"
        with self.assertRaisesRegex(SpecError, "historical holdout"):
            validate_spec(spec)

    def test_rejects_confirmation_without_path_or_hash(self):
        spec = copy.deepcopy(self.specs["grpo.json"])
        spec["data"]["confirmation"].pop("path")
        spec["data"]["confirmation"].pop("sha256")
        with self.assertRaises(SpecError):
            validate_spec(spec)

    def test_rejects_training_entry_without_path_or_hash(self):
        spec = copy.deepcopy(self.specs["cpt.json"])
        spec["data"]["training"] = [{"role": "training"}]
        with self.assertRaisesRegex(SpecError, "sha256"):
            validate_spec(spec)

    def test_rejects_development_without_hash(self):
        spec = copy.deepcopy(self.specs["distillation.json"])
        spec["data"]["development"].pop("sha256")
        with self.assertRaisesRegex(SpecError, "development data lacks hash"):
            validate_spec(spec)

    def test_interpretation_can_describe_evaluation_limitations(self):
        spec = copy.deepcopy(self.specs["synthetic_replacement.json"])
        spec["claim"]["interpretation_if_supported"] = "This is a local comparison, not an external blind test."
        validate_spec(spec)

    def test_design_does_not_require_phase_exit_approval(self):
        spec = copy.deepcopy(self.specs["cpt.json"])
        self.assertNotIn("required_for_phase_exit", spec["external_blind"])
        spec["claim"]["wording_if_inconclusive"] = "本地样本结果不确定，可考虑新的盲测样本。"
        validate_spec(spec)

    def test_rejects_local_candidate_labeled_blind(self):
        spec = copy.deepcopy(self.specs["synthetic_replacement.json"])
        spec["data"]["confirmation"]["is_blind"] = True
        with self.assertRaisesRegex(SpecError, "must not be called blind"):
            validate_spec(spec)

    def test_rejects_missing_starting_weights(self):
        spec = copy.deepcopy(self.specs["dpo_ipo.json"])
        spec["base_model"].pop("weights")
        with self.assertRaisesRegex(SpecError, "weights"):
            validate_spec(spec)

    def test_rejects_config_file_as_starting_weights(self):
        spec = copy.deepcopy(self.specs["dpo_ipo.json"])
        spec["base_model"]["weights"] = {
            **spec["base_model"]["config"],
            "size_bytes": 8192,
        }
        with self.assertRaisesRegex(SpecError, "safetensors"):
            validate_spec(spec)

    def test_rejects_null_grpo_probe_threshold(self):
        spec = copy.deepcopy(self.specs["grpo.json"])
        spec["method_specific"]["reward_hacking_probes"][0]["pass_threshold"] = None
        with self.assertRaisesRegex(SpecError, "threshold invalid"):
            validate_spec(spec)

    def test_rejects_merged_cpt_estimand(self):
        spec = copy.deepcopy(self.specs["cpt.json"])
        spec["comparison"] = {
            "control": "no_cpt", "treatments": ["lora", "full"],
            "treatment_variable": "adaptation_mode", "fixed_controls": ["a", "b", "c", "d"]
        }
        with self.assertRaisesRegex(SpecError, "exposure and adaptation-mode"):
            validate_spec(spec)

    def test_rejects_non_grouped_preference_training(self):
        spec = copy.deepcopy(self.specs["dpo_ipo.json"])
        spec["data"]["training"][0]["path"] = "phase1/data/processed/preference/train_split.jsonl"
        with self.assertRaises(SpecError):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
