#!/usr/bin/env bash
# =============================================================================
# Week17 GRPO 训练编排 (TRL GRPOTrainer, MCQ 答对率 reward)
# =============================================================================
# base = week12_lora_cpt/50_50_fused (与 DPO 同口径, 非 stub 过时的 week11_cpt_pure)
# data = phase1/data/processed/cmexam/train.jsonl (prepare_cmexam.py 产物)
#
# 默认跑 baseline: mcq_base (G=4, beta=0 不加载 ref, loss=dapo, lr=1e-6, LoRA r16/α32)
#   先跑通 baseline, 机制确认后再做 ablation (num_generations / temp / KL / reward) — stretch。
#
# 完成判断: adapter_model.safetensors (TRL save_model 末步固定名)。
# 全程离线 (国内 HF phone-home 会 hang); MPS 兜底 env 在 train_grpo.py 顶部已设。
#
# 用法:
#   bash phase1/week17/run_grpo.sh                                   # baseline (后台长跑)
#   bash phase1/week17/run_grpo.sh mcq_base ""                       # 显式 baseline
#   bash phase1/week17/run_grpo.sh mcq_g8 "--num-generations 8"      # ablation: G=8
#   bash phase1/week17/run_grpo.sh mcq_beta03 "--kl-coeff 0.3"       # ablation: 开 KL
#   bash phase1/week17/run_grpo.sh mcq_grpo "--loss-type grpo"       # ablation: 原版 loss
#
# 后台跑 (detached, 防 sleep):
#   nohup bash phase1/week17/run_grpo.sh > phase1/results/week17_grpo/grpo_run.log 2>&1 &
#   caffeinate -dimsu -w $! &    # 防系统睡眠
# =============================================================================
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1
PY=phase1/.venv/bin/python
BASE=phase1/results/week12_lora_cpt/50_50_fused
DATA=phase1/data/processed/cmexam/grpo_train.jsonl
SWEEP=phase1/results/week17_grpo
LOG=$SWEEP/grpo_run.log
mkdir -p "$SWEEP"

# run name + 额外 train_grpo.py flags (默认 baseline: 无额外 flag)
NAME="${1:-mcq_base}"
EXTRA="${2:-}"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

out=$SWEEP/$NAME
if [ -f "$out/adapter_model.safetensors" ]; then
  log "==== SKIP $NAME (已训完, marker $out/adapter_model.safetensors) ===="
  exit 0
fi

log "==== Week17 GRPO START (base=50_50_fused, reward=mcq_accuracy, name=$NAME) [$EXTRA] → $out ===="
if $PY -u phase1/week17/train_grpo.py \
      --model "$BASE" --data "$DATA" \
      --output "$out" $EXTRA >>"$LOG" 2>&1; then
  log "✓ TRAIN $NAME OK (adapter @ $out)"
else
  log "✗ TRAIN $NAME FAIL (见 $LOG)"
fi
log "==== Week17 GRPO END (eval 由 run_dpo_eval.py / eval_winrate.py + CMExam holdout 接力) ===="
