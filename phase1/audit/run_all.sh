#!/usr/bin/env bash
set -euo pipefail

phase1_python="phase1/.venv/bin/python"
if [[ ! -x "$phase1_python" ]]; then
  echo "FAIL: phase1/.venv is required for tokenizer-based Week 21 audit" >&2
  exit 1
fi

python3 phase1/audit/build_preference_grouped_split.py >/dev/null
python3 phase1/audit/audit_benchmark_splits.py >/dev/null
python3 phase1/audit/build_cmexam_confirmation_candidate.py >/dev/null
python3 phase1/audit/reanalyse_week17_clean.py >/dev/null
python3 phase1/confirmation/build_grpo_probe_fixtures.py >/dev/null
"$phase1_python" phase1/week21/prepare_clean_matched_replacement.py \
  --real-raw phase1/data/processed/cmexam/train.jsonl \
  --synthetic phase1/results/week21_synthetic/data/evol_instruct.jsonl phase1/results/week21_synthetic/data/self_instruct.jsonl \
  --confirmation phase1/data/processed/cmexam/confirmation_candidate_v1.jsonl \
  --control-output phase1/data/processed/confirmation/week21_control_real_clean_v1.jsonl \
  --treatment-output phase1/data/processed/confirmation/week21_treatment_synthetic50_clean_v1.jsonl \
  --audit-output phase1/audit/week21_clean_replacement_audit.json \
  --n-total 2000 --synthetic-fraction 0.5 --real-pool-size 4000 --similarity-threshold 0.78 \
  --tokenizer phase1/results/week12_lora_cpt/50_50_fused --seed 123 >/dev/null
python3 -m unittest discover -s phase1/audit/tests -v
python3 -m unittest discover -s phase1/confirmation/tests -v
python3 -m unittest discover -s phase1/week21/tests -v
python3 phase1/audit/validate_evidence_protocol.py
python3 phase1/confirmation/validate_specs.py
python3 phase1/audit/validate_repository_consistency.py
