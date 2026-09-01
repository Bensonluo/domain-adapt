from __future__ import annotations

import sys
import unittest
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))

from reanalyse_week17_clean import analyse, paired_statistics  # noqa: E402


class Week17ReanalysisTests(unittest.TestCase):
    def test_overlap_is_removed_and_full_effect_is_bounded(self):
        holdout = [{"prompt": f"question {i}\nA. x", "answer": "A"} for i in range(5)]
        training = [{"prompt": "question 1\nA. x"}]
        control = [
            {"gold": "A", "pred": pred} for pred in ("A", "B", "B")
        ]
        treatment = [
            {"gold": "A", "pred": pred} for pred in ("A", "A", "A")
        ]
        result = analyse(
            holdout, training, control, treatment,
            {"n": 5, "correct": 2, "accuracy": 0.4},
            {"n": 5, "correct": 4, "accuracy": 0.8},
            1000, 7,
        )
        self.assertEqual(result["training_overlap"]["overlap_indices_zero_based"], [1])
        self.assertEqual(result["retained_prefix_reanalysis"]["n"], 2)
        self.assertEqual(result["retained_prefix_reanalysis"]["delta"], 0.5)
        self.assertEqual(result["full_500_clean_effect_partial_identification"]["clean_treatment_minus_control_correct_count_bounds"], [1, 3])

    def test_mcnemar_no_discordance(self):
        stats = paired_statistics([True, False], [True, False], 100, 1)
        self.assertEqual(stats["mcnemar_exact"]["p_value"], 1.0)

    def test_gold_misalignment_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "align"):
            analyse(
                [{"prompt": "q", "answer": "A"}], [],
                [{"gold": "B", "pred": "B"}], [{"gold": "A", "pred": "A"}],
                {"n": 1, "correct": 0, "accuracy": 0.0},
                {"n": 1, "correct": 1, "accuracy": 1.0}, 100, 1,
            )


if __name__ == "__main__":
    unittest.main()
