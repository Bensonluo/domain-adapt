#!/usr/bin/env bash
# =============================================================================
# 修 100-0: 上次 session 杀在 ~1060/2500 iter, adapters.safetensors 是周期 checkpoint
# (mlx_lm 每 100 步存一次), 误触发 run_lora_sweep.sh 的 skip-if-exists → 100-0 欠训练。
# 本脚本等主 sweep 跑完, 清掉 partial 100-0, 重训到 2500, 再重跑 fuse/eval/summary。
# detached (nohup), 不打断正在跑的 70-30/50-50, session 退出也不死。
# =============================================================================
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MTL_TIMEOUT=0
PY=.venv/bin/python
SWEEP=phase1/results/week12_lora_cpt
LOG=$SWEEP/sweep_run.log
log(){ printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

log "==== FIX-100-0 waiter: 等主 sweep (run_lora_sweep.sh) 跑完 ===="
while pgrep -f "run_lora_sweep.sh" >/dev/null; do sleep 60; done
log "==== 主 sweep 已结束; 开始重训 100-0 到 2500 ===="

# 清 partial 100-0 (只到 ~1060 iter, 不公平对照) + 旧 fused (强制重新 fuse)
rm -rf "$SWEEP/100_0/adapters" "$SWEEP/100_0_fused"
log "已清 partial 100-0 artifacts, 重训中..."

if $PY phase1/week11/train_cpt.py --model Qwen/Qwen3-1.7B --ratio 100-0 \
      --fine-tune-type lora --config phase1/week11/lora_cpt_config.yaml \
      --lr 1e-5 --iters 2500 --batch-size 1 \
      --output "$SWEEP/100_0" --no-generate >>"$LOG" 2>&1; then
  log "✓ 100-0 重训到 2500 完成"
else
  log "✗ 100-0 重训 FAIL (见 $LOG)"; exit 1
fi

# 重跑 fuse+eval (70/30,50/50 的 fused 已存在→fuse 幂等跳过; eval 固定 seed 重跑无害) + summary
log "==== 重跑 fuse + eval + summary ===="
$PY phase1/week12/run_lora_sweep_eval.py >>"$LOG" 2>&1 && log "✓ eval OK" || log "✗ eval FAIL"
$PY phase1/week12/lora_sweep_summary.py >>"$LOG" 2>&1 || log "✗ summary FAIL"
log "==== FIX-100-0 全部完成 — 看 sweep_summary.json ===="
