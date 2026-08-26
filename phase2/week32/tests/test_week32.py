"""Regression tests for the Week 32 scaffold and evidence gate."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
VALIDATOR = WEEK_DIR / "validate_week32.py"


class Week32Tests(unittest.TestCase):
    def test_code_scaffold_is_valid(self) -> None:
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--scope", "code"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_complete_gate_fails_until_real_deliverables_exist(self) -> None:
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--scope", "complete"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
