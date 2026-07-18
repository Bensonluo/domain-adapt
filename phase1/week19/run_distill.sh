#!/usr/bin/env bash
# =============================================================================
# Week19 Response-Distillation 三对照编排 (TRL SFTTrainer + PEFT-LoRA)
# =============================================================================
# base  = week12_lora_cpt/50_50_fused (与 DPO/GRPO 同口径, delta 可比)
# 三臂 (同 2000 CMExam 题, 只换 completion 来源):
#   real   = 人写 {Answer}\n{Explanation}        (train_distill_sft.py, 已单独跑)
#   distill= Qwen3-30B-A3B teacher 答案 (mlx_lm direct, greedy)
#   mixed  = 1000 real + 1000 distill
#
# 流程 (幂等, 失败可重跑):
#   1. train distill + mixed (real 已训; adapter_model.safetensors 为完成 marker)
#   2. run_dpo_eval.py --runs real distill mixed --skip-base  → PEFT merge 各臂 + CMMLU
#      (复用 week17 base_hf base scores, 拷过来, 同 base 同口径)
#   3. eval_cmexam.py × 3 (CMExam holdout 500, 同 week17 口径)
#   4. summarize_distill.py → distill_summary.json + 三对照表 (含 GRPO 对照)
#
# 用法:
#   bash phase1/week19/run_distill.sh                       # 全流程 (后台长跑)
#   nohup bash phase1/week19/run_distill.sh > .../run.log 2>&1 &
# =============================================================================
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
PY=phase1/.venv/bin/python
BASE=phase1/results/week12_lora_cpt/50_50_fused
SWEEP=phase1/results/week19_distill
DATA=$SWEEP/data
LOG=$SWEEP/run.log
mkdir -p "$SWEEP"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# ── 0. base scores + base CMExam holdout (复用 week17, 同 base 同口径) ──
mkdir -p "$SWEEP/base_hf"
if [ ! -f "$SWEEP/base_hf/scores_50_50_fused.json" ]; then
  if [ -f phase1/results/week17_grpo/base_hf/scores_50_50_fused.json ]; then
    cp phase1/results/week17_grpo/base_hf/scores_50_50_fused.json "$SWEEP/base_hf/"
    log "拷 week17 base CMMLU scores → $SWEEP/base_hf/"
  fi
fi
if [ ! -f "$SWEEP/base_cmexam_holdout.json" ] && [ -f phase1/results/week17_grpo/base_cmexam_holdout.json ]; then
  cp phase1/results/week17_grpo/base_cmexam_holdout.json "$SWEEP/"
  log "拷 week17 base CMExam holdout → $SWEEP/base_cmexam_holdout.json"
fi

# ── 1. train distill + mixed (real 已单独训) ──
for v in distill mixed; do
  out=$SWEEP/$v
  if [ -f "$out/adapter_model.safetensors" ]; then
    log "==== SKIP train $v (已训完) ===="
    continue
  fi
  if [ ! -f "$DATA/${v}_sft.jsonl" ]; then
    log "==== [!] 跳过 $v: 数据 $DATA/${v}_sft.jsonl 不存在 (teacher 跑完 + Phase B 后才有) ===="
    continue
  fi
  log "==== TRAIN $v → $out ===="
  $PY -u phase1/week19/train_distill_sft.py \
      --model "$BASE" --data "$DATA/${v}_sft.jsonl" --output "$out" \
      --lr 2e-5 --epochs 3 --batch-size 4 --grad-accum 4 --max-length 1536 >>"$LOG" 2>&1 \
    && log "✓ TRAIN $v OK" || log "✗ TRAIN $v FAIL"
done

# ── 2. CMMLU eval (PEFT merge 各臂 + medical_cn/general_cn, --skip-base) ──
log "==== CMMLU eval (run_dpo_eval, 三臂 merge+eval) ===="
$PY -u phase1/week15/run_dpo_eval.py \
    --sweep "$SWEEP" --base "$BASE" --runs real distill mixed --skip-base >>"$LOG" 2>&1 \
  && log "✓ CMMLU eval OK" || log "✗ CMMLU eval FAIL (部分臂可能未训完)"

# ── 3. CMExam holdout × 3 (fused 已由 run_dpo_eval 产出) ──
for v in real distill mixed; do
  fused=$SWEEP/${v}_fused
  if [ ! -f "$fused/config.json" ]; then
    log "==== [!] 跳过 CMExam $v: fused 不存在 ($fused) ===="; continue
  fi
  if [ -f "$fused/cmexam_holdout.json" ]; then
    log "==== SKIP CMExam $v (已评) ===="; continue
  fi
  log "==== CMExam holdout $v ===="
  $PY -u phase1/week17/eval_cmexam.py \
      --model "$fused" --output "$fused/cmexam_holdout.json" >>"$LOG" 2>&1 \
    && log "✓ CMExam $v OK" || log "✗ CMExam $v FAIL"
done

# ── 4. 汇总 ──
log "==== SUMMARIZE ===="
$PY -u phase1/week19/summarize_distill.py --sweep "$SWEEP" --runs real distill mixed >>"$LOG" 2>&1 \
  && log "✓ 全流程完成 → $SWEEP/distill_summary.json" || log "✗ SUMMARIZE FAIL"
