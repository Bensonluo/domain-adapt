from __future__ import annotations

import unittest

from adaptstack.core.registry import StageRegistry


class RegistryTests(unittest.TestCase):
    def test_duplicate_registration_fails(self) -> None:
        registry = StageRegistry()
        registry.register("example", lambda: object())  # type: ignore[arg-type,return-value]
        with self.assertRaises(ValueError):
            registry.register("example", lambda: object())  # type: ignore[arg-type,return-value]

    def test_unknown_stage_fails(self) -> None:
        with self.assertRaisesRegex(KeyError, "no implementation"):
            StageRegistry().create("missing")

    def test_path_like_stage_name_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            StageRegistry().register("../outside", lambda: object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
