from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return environment

    def test_doctor(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--doctor"],
            cwd=ROOT,
            env=self._environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(item["ok"] for item in json.loads(result.stdout).values()))

    def test_legal_plan(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", "configs/legal.yaml", "--show-plan"],
            cwd=ROOT,
            env=self._environment(),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training.dpo", json.loads(result.stdout)["stages"])

    def test_medical_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    "configs/medical.yaml",
                    "--dry-run",
                    "--output",
                    temp_dir,
                ],
                cwd=ROOT,
                env=self._environment(),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run_dir = Path(json.loads(result.stdout)["run_dir"])
            self.assertTrue((run_dir / "run_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
