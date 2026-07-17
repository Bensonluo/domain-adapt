#!/usr/bin/env bash
# =============================================================================
# Week12 LoRA-CPT 比例 sweep (Qwen3-1.7B) — 串行训练 + fuse/eval + summary
# =============================================================================
# 设计: 三比例 (100-0/70-30/50-50) 只变数据比例, 其余全固定 (公平对照):
#   model=Qwen/Qwen3-1.7B  fine-tune-type=lora  config=lora_cpt_config.yaml
#   lr=1e-5  batch-size=1  iters=2500 (跨比例相等 → 比例是唯一变量)
# batch=1: 经 smoke + probe 确认稳 (2.6s/iter, 6.7GB); batch=4 不更快 (带宽受限), 故用 batch=1。
#
# 流程: train×3 → run_lora_sweep_eval.py (fuse+eval) → lora_sweep_summary.py (gain/遗忘/选优)
# 单比例失败不中断整体 (记录 FAIL, 继续); 全程离线 + MTL_TIMEOUT=0 防 GPU hang。
#
# 用法: bash phase1/week12/run_lora_sweep.sh   (后台长跑 ~6h)
# =============================================================================
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MTL_TIMEOUT=0
PY=.venv/bin/python
CFG=phase1/week11/lora_cpt_config.yaml
ITERS=2500
SWEEP=phase1/results/week12_lora_cpt
LOG=$SWEEP/sweep_run.log
mkdir -p "$SWEEP"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

log "==== LoRA-CPT sweep START (Qwen3-1.7B, iters=$ITERS, batch=1, lr=1e-5) ===="

for ratio in 100-0 70-30 50-50; do
  tag=${ratio//-/_}
  out=$SWEEP/$tag
  # 完成判断用 iter=$ITERS 的周期 checkpoint (mlx_lm 每 100 步存 0000NNN_adapters.safetensors),
  # 不能用 adapters.safetensors —— 那个是"最新一次周期保存", iter 100 就有, 会误判 partial 为完成。
  done_marker="$out/adapters/$(printf '%07d' "$ITERS")_adapters.safetensors"
  if [ -f "$done_marker" ]; then
    log "---- SKIP ratio=$ratio (已训完 $ITERS iter, marker $done_marker) ----"
    continue
  fi
  log "---- TRAIN ratio=$ratio → $out ----"
  if $PY phase1/week11/train_cpt.py \
        --model Qwen/Qwen3-1.7B \
        --ratio "$ratio" \
        --fine-tune-type lora \
        --config "$CFG" \
        --lr 1e-5 \
        --iters $ITERS \
        --batch-size 1 \
        --output "$out" \
        --no-generate >>"$LOG" 2>&1; then
    log "✓ TRAIN $ratio OK (adapters @ $out/adapters)"
  else
    log "✗ TRAIN $ratio FAIL (见 $LOG)"
  fi
done

log "---- FUSE + EVAL (run_lora_sweep_eval.py) ----"
if $PY phase1/week12/run_lora_sweep_eval.py >>"$LOG" 2>&1; then
  log "✓ FUSE+EVAL OK"
else
  log "✗ FUSE+EVAL FAIL"
fi

log "---- SUMMARY (lora_sweep_summary.py) ----"
$PY phase1/week12/lora_sweep_summary.py >>"$LOG" 2>&1 || log "✗ SUMMARY FAIL"

log "==== LoRA-CPT sweep END ===="
