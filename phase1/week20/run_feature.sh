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
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
PY=phase1/.venv/bin/python
BASE=phase1/results/week12_lora_cpt/50_50_fused
SWEEP=phase1/results/week20_distill
DATA=$SWEEP/data
LOGITS=$DATA/teacher_topk_logits.jsonl
SFT=phase1/results/week19_distill/data/distill_sft.jsonl
LOG=$SWEEP/run_feature.log
mkdir -p "$SWEEP" "$DATA"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# ── 0. 前置检查 + base scores 复用 ──
[ -f "$LOGITS" ] || { log "✗ 缺 $LOGITS (先跑 extract_teacher_logits.py --resume)"; exit 1; }
[ -f "$SFT" ]    || { log "✗ 缺 $SFT"; exit 1; }
NLOGITS=$(grep -c . "$LOGITS" 2>/dev/null || echo 0)
log "teacher_logits=$NLOGITS 条 (需 2000)"

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
  if [ -f "$out/adapter_model.safetensors" ]; then
    log "==== SKIP train $v (已训完) ===="
    continue
  fi
  log "==== TRAIN $v (α=$alpha T=$temp) → $out ===="
  $PY -u phase1/week20/train_logit_kd.py \
      --model "$BASE" --data "$SFT" --logits "$LOGITS" --output "$out" \
      --alpha "$alpha" --temperature "$temp" --topk 20 \
      --lr 2e-5 --epochs 3 --batch-size 4 --grad-accum 4 --max-length 1536 >>"$LOG" 2>&1 \
    && log "✓ TRAIN $v OK" || log "✗ TRAIN $v FAIL"
done

# ── 2. CMMLU eval (PEFT merge 各臂 + medical_cn/general_cn, --skip-base) ──
log "==== CMMLU eval (run_dpo_eval, 三臂 merge+eval) ===="
$PY -u phase1/week15/run_dpo_eval.py \
    --sweep "$SWEEP" --base "$BASE" --runs kd_t2 kd_t5 kd_pure --skip-base >>"$LOG" 2>&1 \
  && log "✓ CMMLU eval OK" || log "✗ CMMLU eval FAIL (部分臂可能未训完)"

# ── 3. CMExam holdout × 3 (fused 已由 run_dpo_eval 产出) ──
for v in kd_t2 kd_t5 kd_pure; do
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

# ── 4. 汇总 (Part A; Part B 后再跑一次并入) ──
log "==== SUMMARIZE (Part A) ===="
$PY -u phase1/week20/summarize_week20.py --sweep "$SWEEP" --runs kd_t2 kd_t5 kd_pure >>"$LOG" 2>&1 \
  && log "✓ Part A 全流程完成 → $SWEEP/week20_summary.json" || log "✗ SUMMARIZE FAIL (summarize_week20.py 未就绪?)"
