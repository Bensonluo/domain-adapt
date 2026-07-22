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
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
PY=phase1/.venv/bin/python
PYTX=/Users/luopeng/Documents/GitHub/4bit-QLoRA-post-training/venv/bin/python
TEACHER=~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit
BASE=phase1/results/week12_lora_cpt/50_50_fused
SWEEP=phase1/results/week20_distill
DATA=$SWEEP/data
SFT=phase1/results/week19_distill/data/distill_sft.jsonl
LOG=$SWEEP/run_onpolicy.log
mkdir -p "$SWEEP" "$DATA"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# ── 1. student on-policy generate (N=8, resume) ──
SAMPLES=$DATA/student_samples.jsonl
NSAMP=$(grep -c . "$SAMPLES" 2>/dev/null || echo 0)
if [ "$NSAMP" -ge 2000 ]; then
  log "==== SKIP generate ($NSAMP 题已采) ===="
else
  log "==== GENERATE student N=8 ($NSAMP/2000, resume) → $SAMPLES ===="
  $PY -u phase1/week20/generate_student_samples.py \
      --model "$BASE" --data "$SFT" --out "$SAMPLES" \
      --n 8 --temperature 0.8 --top-p 0.95 --max-new-tokens 128 --resume >>"$LOG" 2>&1 \
    && log "✓ GENERATE OK" || log "✗ GENERATE FAIL"
fi

# ── 2. teacher judge 1-5 (resume) ──
SCORES=$DATA/judge_scores.jsonl
NSCORE=$(grep -c . "$SCORES" 2>/dev/null || echo 0)
# 2000 题 × 8 = 16000
if [ "$NSCORE" -ge 16000 ]; then
  log "==== SKIP judge ($NSCORE 已评) ===="
else
  log "==== JUDGE teacher 1-5 ($NSCORE/16000, resume) → $SCORES ===="
  $PYTX -u phase1/week20/judge_with_teacher.py \
      --teacher "$TEACHER" --samples "$SAMPLES" --out "$SCORES" --resume >>"$LOG" 2>&1 \
    && log "✓ JUDGE OK" || log "✗ JUDGE FAIL"
fi

# ── 3. prepare best-of-N → 4 臂数据 ──
log "==== PREPARE on-policy data ===="
$PY -u phase1/week20/prepare_onpolicy_data.py \
    --samples "$SAMPLES" --scores "$SCORES" --out-dir "$DATA" >>"$LOG" 2>&1 \
  && log "✓ PREPARE OK" || log "✗ PREPARE FAIL"

# ── 4. train 3 SFT 臂 (week19 train_distill_sft.py, 同 lr/ep/b/ga) ──
for v in rs_mcq rs_teacher rs_both; do
  out=$SWEEP/$v
  if [ -f "$out/adapter_model.safetensors" ]; then
    log "==== SKIP train $v (已训完) ===="; continue
  fi
  if [ ! -f "$DATA/${v}_sft.jsonl" ]; then
    log "==== [!] 跳过 $v: 数据缺 ===="; continue
  fi
  log "==== TRAIN $v → $out ===="
  $PY -u phase1/week19/train_distill_sft.py \
      --model "$BASE" --data "$DATA/${v}_sft.jsonl" --output "$out" \
      --lr 2e-5 --epochs 3 --batch-size 4 --grad-accum 4 --max-length 1536 >>"$LOG" 2>&1 \
    && log "✓ TRAIN $v OK" || log "✗ TRAIN $v FAIL"
done

# ── 5. CMMLU eval (PEFT merge 三臂 + medical_cn/general_cn) ──
log "==== CMMLU eval (run_dpo_eval, 三臂 merge+eval) ===="
$PY -u phase1/week15/run_dpo_eval.py \
    --sweep "$SWEEP" --base "$BASE" --runs rs_mcq rs_teacher rs_both --skip-base >>"$LOG" 2>&1 \
  && log "✓ CMMLU eval OK" || log "✗ CMMLU eval FAIL"

# ── 6. CMExam holdout × 3 ──
for v in rs_mcq rs_teacher rs_both; do
  fused=$SWEEP/${v}_fused
  if [ ! -f "$fused/config.json" ]; then
    log "==== [!] 跳过 CMExam $v: fused 不存在 ===="; continue
  fi
  if [ -f "$fused/cmexam_holdout.json" ]; then
    log "==== SKIP CMExam $v (已评) ===="; continue
  fi
  log "==== CMExam holdout $v ===="
  $PY -u phase1/week17/eval_cmexam.py \
      --model "$fused" --output "$fused/cmexam_holdout.json" >>"$LOG" 2>&1 \
    && log "✓ CMExam $v OK" || log "✗ CMExam $v FAIL"
done

# ── 7. 汇总 (Part A + B 合并) ──
log "==== SUMMARIZE (Part A + B) ===="
$PY -u phase1/week20/summarize_week20.py \
    --sweep "$SWEEP" --runs kd_t2 kd_t5 kd_pure rs_mcq rs_teacher rs_both >>"$LOG" 2>&1 \
  && log "✓ 全流程完成 → $SWEEP/week20_summary.json" || log "✗ SUMMARIZE FAIL"
