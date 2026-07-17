#!/usr/bin/env bash
# =============================================================================
# Week15 DPO β-sweep (TRL + PyTorch/MPS + PEFT-LoRA) — 3 个 β 串行训练
# =============================================================================
# 设计: 三 β (0.1/0.3/0.5) 只变 β, 其余全固定 (公平对照对齐强度 trade-off):
#   base=week12_lora_cpt/50_50_fused  LoRA r16/alpha32/dropout0.05  loss=sigmoid
#   lr=5e-6  epochs=1  batch=1  max_length=2048  limit=300(子集先定方向)
# 单 β 失败不中断整体 (记录 FAIL, 继续); 全程离线 (国内 HF phone-home 会 hang)。
#
# 完成判断: TRL save_model 末步写 adapter_model.safetensors (固定名, 只在最后存一次),
# 故 [ -f adapter_model.safetensors ] 是干净的 completion marker (无 week12 那种周期 checkpoint 歧义)。
#
# 用法: bash phase1/week15/run_dpo_sweep.sh   (后台长跑 ~2h; 外层 nohup + caffeinate 防睡眠)
# =============================================================================
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
# 国内 HF 网络会 hang (transformers/trl 启动 phone home), 全程离线 (train_dpo.py 内也有 setdefault)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=phase1/.venv/bin/python
BASE=phase1/results/week12_lora_cpt/50_50_fused
DATA=phase1/data/processed/preference/train_split.jsonl
SWEEP=phase1/results/week15_dpo
LOG=$SWEEP/sweep_run.log
mkdir -p "$SWEEP"

LIMIT=300      # 子集先定 β 方向 (用户授权); 胜出者后续可全量重训
MAXLEN=2048    # 统一化 step 时间 ~8-10s (smoke 实测 4096 下 step 18-30s)
LR=5e-6        # LoRA-DPO 比 full-param 5e-7 高一个量级
EPOCHS=1

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

log "==== DPO β-sweep START (base=50_50_fused, limit=$LIMIT, max_len=$MAXLEN, lr=$LR, epochs=$EPOCHS) ===="

for beta in 0.1 0.3 0.5; do
  out=$SWEEP/beta_$beta
  if [ -f "$out/adapter_model.safetensors" ]; then
    log "---- SKIP beta=$beta (已训完, marker $out/adapter_model.safetensors) ----"
    continue
  fi
  log "---- TRAIN beta=$beta → $out ----"
  if $PY -u phase1/week15/train_dpo.py \
        --model "$BASE" \
        --data "$DATA" \
        --beta "$beta" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --limit "$LIMIT" \
        --max-length "$MAXLEN" \
        --output "$out" >>"$LOG" 2>&1; then
    log "✓ TRAIN beta=$beta OK (adapter @ $out)"
  else
    log "✗ TRAIN beta=$beta FAIL (见 $LOG)"
  fi
done

log "==== DPO β-sweep END (3 β 训练完; eval/winrate 由 run_dpo_eval.py + eval_winrate.py 接力) ===="
