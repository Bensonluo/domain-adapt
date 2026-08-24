#!/usr/bin/env bash
# =============================================================================
# Week20 Part B: On-Policy Distillation 编排 (rejection-sampling SFT + teacher judge)
# =============================================================================
# base   = week12_lora_cpt/50_50_fused (与 Part A / week19 同口径)
# student on-policy: N=8/question 采样 → teacher judge 1-5 → best-of-N SFT
#
# 三 SFT 臂 (同 2000 CMExam 题, 只换选择信号):
#   rs_mcq    = rule correctness (letter==gold, = GRPO 同信号) — 全错回退 teacher
#   rs_teacher= teacher judge score (盲评软信号)
#   rs_both   = correct ∩ teacher 精排 (rule 兜底 + teacher 精排)
# (dpo_onpolicy 为 stretch, 数据已生成, 训练另行 week15 DPO 栈, 此脚本不自动跑)
#
# 两 venv:
#   PY = phase1/.venv (student generate / SFT train / eval, HF transformers+MPS)
#   PYTX = 4bit-QLoRA-post-training venv (teacher judge, mlx_lm)
#
# 流程 (幂等, 失败可重跑):
#   1. generate student N=8 (resume)
#   2. judge teacher 1-5 (resume)
#   3. prepare_onpolicy_data → 3 SFT jsonl + dpo jsonl
#   4. train 3 SFT 臂 (week19 train_distill_sft.py; adapter_model.safetensors 为 marker)
#   5. run_dpo_eval --runs rs_mcq rs_teacher rs_both --skip-base → PEFT merge + CMMLU
#   6. eval_cmexam × 3
#   7. summarize_week20 (Part B; 与 Part A 合并)
#
# 用法:
#   bash phase1/week20/run_onpolicy.sh
#   nohup bash phase1/week20/run_onpolicy.sh > .../run_onpolicy.log 2>&1 &
# =============================================================================
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
PY=${WEEK20_PYTHON:-phase1/.venv/bin/python}
PYTX=${WEEK20_TEACHER_PYTHON:-/Users/luopeng/Documents/GitHub/4bit-QLoRA-post-training/venv/bin/python}
TEACHER=${WEEK20_TEACHER:-${HOME}/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit}
BASE=${WEEK20_BASE:-phase1/results/week12_lora_cpt/50_50_fused}
SWEEP=${WEEK20_SWEEP:-phase1/results/week20_distill}
LINEAGE=${WEEK20_LINEAGE:-phase1/week20/week20_lineage.json}
DATA=$SWEEP/data
SFT=phase1/results/week19_distill/data/distill_sft.jsonl
LOG=$SWEEP/run_onpolicy.log
mkdir -p "$SWEEP" "$DATA"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# ── 1. student on-policy generate (N=8, resume) ──
SAMPLES=$DATA/student_samples.jsonl
SAMPLES_STRUCTURAL=0
SAMPLES_INPUT_OK=0
if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --scope samples --skip-lineage >/dev/null 2>&1; then
  SAMPLES_STRUCTURAL=1
fi
if "$PY" phase1/week20/artifact_lineage.py verify --manifest "$LINEAGE" \
    --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
    --stage samples --inputs-only >/dev/null 2>&1; then
  SAMPLES_INPUT_OK=1
fi
if [ "$SAMPLES_STRUCTURAL" -eq 1 ] \
    && "$PY" phase1/week20/artifact_lineage.py verify --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
      --stage samples >>"$LOG" 2>&1; then
  log "==== SKIP generate (2,000 题 × 8 已验证) ===="
else
  log "==== GENERATE student N=8 (resume) → $SAMPLES ===="
  GEN_RESUME=(--resume)
  if [ -f "$SAMPLES" ] && [ "$SAMPLES_INPUT_OK" -eq 0 ]; then
    quarantine=$DATA/.student_samples.stale-$$.jsonl
    mv "$SAMPLES" "$quarantine"
    GEN_RESUME=()
    log "旧 samples lineage 不匹配，已移至 $quarantine (可恢复)"
  fi
  if "$PY" -u phase1/week20/generate_student_samples.py \
      --model "$BASE" --data "$SFT" --out "$SAMPLES" \
      --n 8 --temperature 0.8 --top-p 0.95 --max-new-tokens 128 "${GEN_RESUME[@]}" >>"$LOG" 2>&1; then
    log "✓ GENERATE OK"
  else
    log "✗ GENERATE FAIL"
    exit 1
  fi
  "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --scope samples >>"$LOG" 2>&1 \
    || { log "✗ student samples 完整性验证失败"; exit 1; }
  "$PY" phase1/week20/artifact_lineage.py record --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
      --stage samples >>"$LOG" 2>&1
fi

# ── 2. teacher judge 1-5 (resume) ──
SCORES=$DATA/judge_scores.jsonl
SCORES_STRUCTURAL=0
SCORES_INPUT_OK=0
if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --scope scores --skip-lineage >/dev/null 2>&1; then
  SCORES_STRUCTURAL=1
fi
if "$PY" phase1/week20/artifact_lineage.py verify --manifest "$LINEAGE" \
    --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
    --stage scores --inputs-only >/dev/null 2>&1; then
  SCORES_INPUT_OK=1
fi
if [ "$SCORES_STRUCTURAL" -eq 1 ] \
    && "$PY" phase1/week20/artifact_lineage.py verify --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
      --stage scores >>"$LOG" 2>&1; then
  log "==== SKIP judge (16,000 个唯一评分已验证) ===="
else
  log "==== JUDGE teacher 1-5 (resume) → $SCORES ===="
  JUDGE_RESUME=(--resume)
  if [ -f "$SCORES" ] && [ "$SCORES_INPUT_OK" -eq 0 ]; then
    quarantine=$DATA/.judge_scores.stale-$$.jsonl
    mv "$SCORES" "$quarantine"
    JUDGE_RESUME=()
    log "旧 scores lineage 不匹配，已移至 $quarantine (可恢复)"
  fi
  if "$PYTX" -u phase1/week20/judge_with_teacher.py \
      --teacher "$TEACHER" --samples "$SAMPLES" --out "$SCORES" "${JUDGE_RESUME[@]}" >>"$LOG" 2>&1; then
    log "✓ JUDGE OK"
  else
    log "✗ JUDGE FAIL"
    exit 1
  fi
  "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --scope scores >>"$LOG" 2>&1 \
    || { log "✗ judge scores 完整性验证失败"; exit 1; }
  "$PY" phase1/week20/artifact_lineage.py record --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
      --stage scores >>"$LOG" 2>&1
fi

# ── 3. prepare best-of-N → 4 臂数据 ──
log "==== PREPARE on-policy data ===="
if "$PY" -u phase1/week20/prepare_onpolicy_data.py \
    --samples "$SAMPLES" --scores "$SCORES" --out-dir "$DATA" >>"$LOG" 2>&1; then
  log "✓ PREPARE OK"
else
  log "✗ PREPARE FAIL"
  exit 1
fi
"$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --scope prepared >>"$LOG" 2>&1 \
  || { log "✗ on-policy 训练数据完整性验证失败"; exit 1; }

# ── 4. train 3 SFT 臂 (week19 train_distill_sft.py, 同 lr/ep/b/ga) ──
for v in rs_mcq rs_teacher rs_both; do
  out=$SWEEP/$v
  if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
      --expected-base "$BASE" --lineage-manifest "$LINEAGE" \
      --teacher "$TEACHER" \
      --scope checkpoint --arms "$v" >/dev/null 2>&1; then
    log "==== SKIP train $v (checkpoint + provenance 已验证) ===="; continue
  fi
  if [ ! -f "$DATA/${v}_sft.jsonl" ]; then
    log "==== [!] 跳过 $v: 数据缺 ===="; continue
  fi
  log "==== TRAIN $v → $out ===="
  if "$PY" -u phase1/week19/train_distill_sft.py \
      --model "$BASE" --data "$DATA/${v}_sft.jsonl" --output "$out" \
      --lr 2e-5 --epochs 3 --batch-size 4 --grad-accum 4 --max-length 1536 >>"$LOG" 2>&1; then
    log "✓ TRAIN $v OK"
  else
    log "✗ TRAIN $v FAIL"
    exit 1
  fi
  "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
      --expected-base "$BASE" --lineage-manifest "$LINEAGE" --skip-lineage \
      --scope checkpoint --arms "$v" >>"$LOG" 2>&1 \
    || { log "✗ TRAIN $v 产物完整性验证失败"; exit 1; }
  old_fused=$SWEEP/${v}_fused
  if [ -d "$old_fused" ]; then
    quarantine=$SWEEP/.${v}_fused.pre-retrain-$$
    mv "$old_fused" "$quarantine"
    log "旧 fused/eval 已移至 $quarantine (可恢复)，将从新 adapter 重建"
  fi
  "$PY" phase1/week20/artifact_lineage.py record --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --arms "$v" --stage checkpoint >>"$LOG" 2>&1
done

# ── 5. CMMLU eval (PEFT merge 三臂 + medical_cn/general_cn) ──
log "==== CMMLU eval (run_dpo_eval, 三臂 merge+eval) ===="
if "$PY" -u phase1/week20/run_cmmlu_eval.py \
    --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --manifest "$LINEAGE" \
    --runs rs_mcq rs_teacher rs_both >>"$LOG" 2>&1; then
  log "✓ CMMLU eval OK"
else
  log "✗ CMMLU eval FAIL"
  exit 1
fi

# ── 6. CMExam holdout × 3 ──
for v in rs_mcq rs_teacher rs_both; do
  fused=$SWEEP/${v}_fused
  if [ ! -f "$fused/config.json" ]; then
    log "==== [!] 跳过 CMExam $v: fused 不存在 ===="; continue
  fi
  if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
      --expected-base "$BASE" --lineage-manifest "$LINEAGE" \
      --teacher "$TEACHER" \
      --scope evaluation --arms "$v" >/dev/null 2>&1; then
    log "==== SKIP CMExam $v (fused + eval 已验证) ===="; continue
  fi
  log "==== CMExam holdout $v ===="
  if "$PY" -u phase1/week17/eval_cmexam.py \
      --model "$fused" --output "$fused/cmexam_holdout.json" >>"$LOG" 2>&1; then
    log "✓ CMExam $v OK"
  else
    log "✗ CMExam $v FAIL"
    exit 1
  fi
  "$PY" phase1/week20/artifact_lineage.py record --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --arms "$v" --stage evaluation >>"$LOG" 2>&1
  "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
      --expected-base "$BASE" --lineage-manifest "$LINEAGE" \
      --teacher "$TEACHER" \
      --scope evaluation --arms "$v" >>"$LOG" 2>&1 \
    || { log "✗ CMExam $v 产物完整性验证失败"; exit 1; }
done

# ── 7. 汇总 (Part A + B 合并) ──
log "==== SUMMARIZE (Part A + B) ===="
if "$PY" -u phase1/week20/summarize_week20.py \
    --sweep "$SWEEP" --runs kd_t2 kd_t5 kd_pure rs_mcq rs_teacher rs_both >>"$LOG" 2>&1; then
  log "✓ 汇总完成 → $SWEEP/week20_summary.json"
else
  log "✗ SUMMARIZE FAIL"
  exit 1
fi

if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
    --expected-base "$BASE" --lineage-manifest "$LINEAGE" \
    --teacher "$TEACHER" \
    --scope complete >>"$LOG" 2>&1; then
  log "✓ Week 20 全流程通过完整性验证"
else
  log "✗ Week 20 完整性验证失败 (详见 $LOG)"
  exit 1
fi
