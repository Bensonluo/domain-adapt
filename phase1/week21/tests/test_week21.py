from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

WEEK21 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK21))

from common import normalize_mcq, parse_json_collection  # noqa: E402


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def mcq(index: int):
    return {
        "seed_id": f"seed-{index}", "question": f"测试医学问题编号{index}应选择哪一项",
        "options": [{"key": key, "value": f"选项{key}{index}"} for key in "ABCD"],
        "answer": "A", "explanation": "选项A符合测试医学知识，其他项不符合。",
    }


class Week21Tests(unittest.TestCase):
    def run_script(self, name: str, *args: str):
        completed = subprocess.run(
            [sys.executable, str(WEEK21 / name), *args], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def run_script_fails(self, name: str, *args: str):
        completed = subprocess.run(
            [sys.executable, str(WEEK21 / name), *args], capture_output=True, text=True
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_parser_and_validation(self):
        row, error = normalize_mcq(mcq(1))
        self.assertIsNone(error)
        self.assertEqual(row["answer"], "A")
        self.assertEqual(len(parse_json_collection("```json\n[{\"x\": 1}]\n```")), 1)
        malformed = '[{"question":"足够长的医学测试问题","options":[],"answer":"A","explanation":"说明。“}},' \
                    '{"question":"第二个足够长的医学问题","options":[],"answer":"A","explanation":"说明。"}]'
        self.assertEqual(len(parse_json_collection(malformed)), 2)

    def test_mock_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seeds = root / "seeds.jsonl"
            write_jsonl(seeds, [mcq(index) for index in range(5)])
            out = root / "synthetic"
            self.run_script("self_instruct.py", "--seeds", str(seeds), "--model", "mock", "--backend", "mock", "--n", "8", "--batch-size", "3", "--similarity-threshold", "1", "--output", str(out))
            self_rows = [json.loads(line) for line in (out / "self_instruct.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(self_rows), 8)
            self.assertTrue(all(row["backend"] == "mock" for row in self_rows))
            self.run_script("evol_instruct.py", "--input", str(out / "self_instruct.jsonl"), "--model", "mock", "--backend", "mock", "--depth", "3", "--limit", "6", "--similarity-threshold", "1", "--output", str(out))
            evolved = [json.loads(line) for line in (out / "evol_instruct.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["evolution_depth"] for row in evolved], [1, 2, 3, 1, 2, 3])

            real_raw = root / "real_raw.jsonl"
            write_jsonl(real_raw, [mcq(index + 100) for index in range(10)])
            quality = root / "quality.json"
            audit = root / "audit.jsonl"
            self.run_script("assess_quality.py", "--self-data", str(out / "self_instruct.jsonl"), "--evolved-data", str(out / "evol_instruct.jsonl"), "--real-data", str(real_raw), "--output", str(quality), "--audit-output", str(audit), "--audit-size", "4")
            self.assertEqual(json.loads(quality.read_text())["counts"]["synthetic_total"], 14)
            audit_rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            audit_rows[0]["review_correct"] = True
            audit_rows[0]["review_notes"] = "reviewed"
            audit_rows[0]["reviewer"] = "test reviewer"
            audit_rows[0]["reviewer_kind"] = "human_nonclinician"
            write_jsonl(audit, audit_rows)
            self.run_script("assess_quality.py", "--self-data", str(out / "self_instruct.jsonl"), "--evolved-data", str(out / "evol_instruct.jsonl"), "--real-data", str(real_raw), "--output", str(quality), "--audit-output", str(audit), "--audit-size", "4")
            preserved = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(preserved[0]["review_correct"])
            self.assertEqual(preserved[0]["review_notes"], "reviewed")
            metrics = json.loads(quality.read_text())
            self.assertEqual(metrics["manual_review"]["reviewed"], 1)
            self.assertEqual(metrics["manual_review"]["reviewer_kinds"], ["human_nonclinician"])
            preserved[0]["options"][0]["value"] = "medically changed answer text"
            write_jsonl(audit, preserved)
            self.run_script("assess_quality.py", "--self-data", str(out / "self_instruct.jsonl"), "--evolved-data", str(out / "evol_instruct.jsonl"), "--real-data", str(real_raw), "--output", str(quality), "--audit-output", str(audit), "--audit-size", "4")
            invalidated = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertIsNone(invalidated[0]["review_correct"])

            real_sft = root / "real_sft.jsonl"
            write_jsonl(real_sft, [{"prompt": f"real {i}", "completion": "A\nwhy", "question_id": i, "gold": "A"} for i in range(10)])
            holdout = root / "holdout.jsonl"
            write_jsonl(holdout, [mcq(999)])
            replacement = root / "replacement.jsonl"
            self.run_script("prepare_replacement.py", "--real-sft", str(real_sft), "--synthetic", str(out / "evol_instruct.jsonl"), str(out / "self_instruct.jsonl"), "--holdout", str(holdout), "--output", str(replacement), "--n-total", "8", "--synthetic-fraction", "0.5")
            rows = [json.loads(line) for line in replacement.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(row["source"] == "real" for row in rows), 4)
            self.assertEqual(sum(row["source"] == "synthetic" for row in rows), 4)

    def test_paired_replacement_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control = root / "control.jsonl"
            treatment = root / "treatment.jsonl"
            write_jsonl(control, [
                {"index": i, "gold": "A", "pred": "A" if i < 7 else "B", "correct": i < 7}
                for i in range(10)
            ])
            write_jsonl(treatment, [
                {"index": i, "gold": "A", "pred": "A" if i < 6 else "B", "correct": i < 6}
                for i in range(10)
            ])
            output = root / "statistics.json"
            self.run_script(
                "analyze_replacement.py", "--control", str(control), "--treatment", str(treatment),
                "--output", str(output), "--bootstrap-samples", "1000",
            )
            result = json.loads(output.read_text())
            self.assertAlmostEqual(result["delta"], -0.1)
            self.assertFalse(result["point_estimate_meets_margin"])
            self.assertEqual(result["mcnemar_exact"]["control_only_correct"], 1)

    def test_clean_matched_replacement_is_deterministic_and_leakage_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "real.jsonl"
            synthetic = root / "synthetic.jsonl"
            confirmation = root / "confirmation.jsonl"
            # 40 real rows provide enough same-label choices for matching.
            real_rows = []
            for index in range(40):
                row = mcq(index + 100)
                row["answer"] = "ABCD"[index % 4]
                row["Answer"] = row.pop("answer")
                row["Question"] = row.pop("question")
                row["Options"] = row.pop("options")
                row["Explanation"] = row.pop("explanation")
                real_rows.append(row)
            write_jsonl(real, real_rows)
            synthetic_rows = [mcq(index + 1000) for index in range(5)]
            for index, row in enumerate(synthetic_rows):
                row["answer"] = "ABCD"[index % 4]
                row["synthetic_id"] = f"synthetic-{index}"
                row["method"] = "test"
            # One synthetic row exactly overlaps confirmation and must be rejected.
            synthetic_rows[0] = {**mcq(9999), "synthetic_id": "leaked", "method": "test"}
            write_jsonl(synthetic, synthetic_rows)
            write_jsonl(confirmation, [{"id": "confirm", "prompt": "测试医学问题编号9999应选择哪一项\nA. x\n答案：", "answer": "A"}])

            control = root / "control.jsonl"
            treatment = root / "treatment.jsonl"
            audit = root / "audit.json"
            args = (
                "--real-raw", str(real), "--synthetic", str(synthetic),
                "--confirmation", str(confirmation), "--control-output", str(control),
                "--treatment-output", str(treatment), "--audit-output", str(audit),
                "--n-total", "8", "--synthetic-fraction", "0.5", "--real-pool-size", "40",
            )
            self.run_script("prepare_clean_matched_replacement.py", *args)
            first_control = control.read_bytes()
            first_treatment = treatment.read_bytes()
            self.run_script("prepare_clean_matched_replacement.py", *args)
            self.assertEqual(first_control, control.read_bytes())
            self.assertEqual(first_treatment, treatment.read_bytes())
            report = json.loads(audit.read_text())
            self.assertTrue(report["invariants"]["equal_arm_records"])
            self.assertTrue(report["invariants"]["equal_label_distribution"])
            self.assertEqual(report["invariants"]["control_prohibited_overlap"], 0)
            self.assertEqual(report["invariants"]["treatment_prohibited_overlap"], 0)
            self.assertGreaterEqual(report["filtering"]["synthetic_rejected"].get("confirmation_similarity", 0), 1)

    def test_manual_review_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = Path(temp) / "audit.jsonl"
            write_jsonl(audit, [{"synthetic_id": "unexpected"}])
            self.run_script_fails("apply_audit_labels.py", "--audit", str(audit))
            row = json.loads(audit.read_text())
            self.assertNotIn("review_correct", row)


if __name__ == "__main__":
    unittest.main()
