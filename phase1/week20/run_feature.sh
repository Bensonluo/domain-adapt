#!/usr/bin/env bash
# =============================================================================
# Week20 Part A: Feature Distillation (Logit-KD) 三臂编排
# =============================================================================
# base   = week12_lora_cpt/50_50_fused (与 DPO/GRPO/SFT 同口径, delta 可比)
# logits = extract_teacher_logits.py 产物 (teacher MLX forward top-K raw logits)
# data   = week19 distill_sft.jsonl (★与 week19 distill 臂同题同 completion → 唯一变量 loss)
#
# 三臂 (同 2000 CMExam 题, 只换 KD α/T):
#   kd_t2   = α=0.5 T=2  (经典 Hinton KD)
#   kd_t5   = α=0.5 T=5  (更软, 暴露更多 dark knowledge)
#   kd_pure = α=0   T=2  (纯 KL 不学 hard label)
#   对照 week19 distill (α=1 纯 CE hard label) — 已有结果, summarize 时并入
#
# 流程 (幂等, 失败可重跑):
#   1. train 三臂 (train_logit_kd.py; adapter_model.safetensors 为完成 marker)
#   2. run_dpo_eval.py --runs kd_t2 kd_t5 kd_pure --skip-base → PEFT merge + CMMLU
#      (复用 week17 base scores, 拷过来同 base 同口径)
#   3. eval_cmexam.py × 3 (CMExam holdout 500, 同 week17 口径)
#   4. summarize_week20.py → week20_summary.json (Part A; Part B 后并入)
#
# 用法:
#   bash phase1/week20/run_feature.sh
#   nohup bash phase1/week20/run_feature.sh > .../run_feature.log 2>&1 &
# =============================================================================
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
PY=${WEEK20_PYTHON:-phase1/.venv/bin/python}
BASE=${WEEK20_BASE:-phase1/results/week12_lora_cpt/50_50_fused}
TEACHER=${WEEK20_TEACHER:-${HOME}/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit}
SWEEP=${WEEK20_SWEEP:-phase1/results/week20_distill}
LINEAGE=${WEEK20_LINEAGE:-phase1/week20/week20_lineage.json}
DATA=$SWEEP/data
LOGITS=$DATA/teacher_topk_logits.jsonl
SFT=phase1/results/week19_distill/data/distill_sft.jsonl
LOG=$SWEEP/run_feature.log
mkdir -p "$SWEEP" "$DATA"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# ── 0. 前置检查 + base scores 复用 ──
[ -f "$LOGITS" ] || { log "✗ 缺 $LOGITS (先跑 extract_teacher_logits.py --resume)"; exit 1; }
[ -f "$SFT" ]    || { log "✗ 缺 $SFT"; exit 1; }
log "验证 teacher logits 完整性"
if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
    --scope logits >>"$LOG" 2>&1 \
    && "$PY" phase1/week20/artifact_lineage.py verify --manifest "$LINEAGE" \
      --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --teacher "$TEACHER" \
      --stage logits >>"$LOG" 2>&1; then
  log "✓ teacher logits 完整"
else
  log "✗ teacher logits 不完整或损坏 (详见 $LOG)"
  exit 1
fi

mkdir -p "$SWEEP/base_hf"
if [ ! -f "$SWEEP/base_hf/scores_50_50_fused.json" ] && [ -f phase1/results/week17_grpo/base_hf/scores_50_50_fused.json ]; then
  cp phase1/results/week17_grpo/base_hf/scores_50_50_fused.json "$SWEEP/base_hf/"
  log "拷 week17 base CMMLU scores → $SWEEP/base_hf/"
fi
if [ ! -f "$SWEEP/base_cmexam_holdout.json" ] && [ -f phase1/results/week17_grpo/base_cmexam_holdout.json ]; then
  cp phase1/results/week17_grpo/base_cmexam_holdout.json "$SWEEP/"
  log "拷 week17 base CMExam holdout → $SWEEP/base_cmexam_holdout.json"
fi

# ── 1. train 三臂 (α/T 不同, 其余同 week19 SFT 栈: lr2e-5 / ep3 / b4 / ga4) ──
# 臂定义: name|alpha|temperature
ARMS="kd_t2|0.5|2 kd_t5|0.5|5 kd_pure|0.0|2"
for spec in $ARMS; do
  IFS='|' read -r v alpha temp <<< "$spec"
  out=$SWEEP/$v
  if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
      --expected-base "$BASE" --lineage-manifest "$LINEAGE" \
      --teacher "$TEACHER" \
      --scope checkpoint --arms "$v" >/dev/null 2>&1; then
    log "==== SKIP train $v (checkpoint + provenance 已验证) ===="
    continue
  fi
  log "==== TRAIN $v (α=$alpha T=$temp) → $out ===="
  if "$PY" -u phase1/week20/train_logit_kd.py \
      --model "$BASE" --data "$SFT" --logits "$LOGITS" --output "$out" \
      --alpha "$alpha" --temperature "$temp" --topk 20 \
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

# ── 2. CMMLU eval (PEFT merge 各臂 + medical_cn/general_cn, --skip-base) ──
log "==== CMMLU eval (run_dpo_eval, 三臂 merge+eval) ===="
if "$PY" -u phase1/week20/run_cmmlu_eval.py \
    --sweep "$SWEEP" --base "$BASE" --source-sft "$SFT" --manifest "$LINEAGE" \
    --runs kd_t2 kd_t5 kd_pure >>"$LOG" 2>&1; then
  log "✓ CMMLU eval OK"
else
  log "✗ CMMLU eval FAIL"
  exit 1
fi

# ── 3. CMExam holdout × 3 (fused 已由 run_dpo_eval 产出) ──
for v in kd_t2 kd_t5 kd_pure; do
  fused=$SWEEP/${v}_fused
  if [ ! -f "$fused/config.json" ]; then
    log "==== [!] 跳过 CMExam $v: fused 不存在 ($fused) ===="; continue
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

# ── 4. 汇总 (Part A; Part B 后再跑一次并入) ──
log "==== SUMMARIZE (Part A) ===="
SUMMARY_RUNS=(kd_t2 kd_t5 kd_pure)
if [ -f "$SWEEP/rs_mcq_fused/cmexam_holdout.json" ] \
    && [ -f "$SWEEP/rs_teacher_fused/cmexam_holdout.json" ] \
    && [ -f "$SWEEP/rs_both_fused/cmexam_holdout.json" ]; then
  SUMMARY_RUNS+=(rs_mcq rs_teacher rs_both)
fi
if "$PY" -u phase1/week20/summarize_week20.py \
    --sweep "$SWEEP" --runs "${SUMMARY_RUNS[@]}" >>"$LOG" 2>&1; then
  log "✓ 汇总完成 → $SWEEP/week20_summary.json"
else
  log "✗ SUMMARIZE FAIL"
  exit 1
fi

if "$PY" phase1/week20/validate_week20.py --sweep "$SWEEP" --source-sft "$SFT" \
    --expected-base "$BASE" --lineage-manifest "$LINEAGE" \
    --teacher "$TEACHER" \
    --scope feature-results >>"$LOG" 2>&1; then
  log "✓ Part A 全流程通过完整性验证"
else
  log "✗ Part A 完整性验证失败 (详见 $LOG)"
  exit 1
fi
