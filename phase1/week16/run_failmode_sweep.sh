#!/usr/bin/env bash
# =============================================================================
# Week16 DPO 失败模式 sweep (从 week15 sigmoid β=0.3 baseline 分支, 每次只改一个变量)
# =============================================================================
# 控制结构: 所有 run 共享 base=50_50_fused + limit=300 + epochs=1 + max_len=2048
#   + lr=5e-6 + LoRA r16/α32, 只在 {noise, β, loss_type} 上各改一个 → 相对 baseline 干净 Δ。
#   baseline = week15 sigmoid β=0.3 (noise=0), 已有产物, 本 sweep 不重跑, compare 阶段直接读。
#
# 6 个 run:
#   noise_0.1/0.3/0.5 : β=0.3 sigmoid + 注入噪声 (dose-response: acc 能否仍冲 1.0 / holdout WR 随剂量降)
#   beta_0.01 / beta_10 : 极端 β (0.01→近无 KL 漂移大或不学; 10→强锚定几乎不动)
#   ipo_0.3          : β=0.3 loss=ipo (length-norm, 攻 week15 sum-WR≈0 长度偏差)
#
# 单 run 失败不中断 (记 FAIL 继续); 全程离线 (国内 HF phone-home 会 hang)。
# 完成判断: adapter_model.safetensors (TRL save_model 末步固定名)。
#
# 用法:
#   bash phase1/week16/run_failmode_sweep.sh                    # 全 6 (后台长跑 ~9h)
#   bash phase1/week16/run_failmode_sweep.sh noise_0.3          # smoke 单跑 (~1.5h)
#   bash phase1/week16/run_failmode_sweep.sh noise_0.1 noise_0.5 beta_0.01 beta_10 ipo_0.3  # 剩 5
# =============================================================================
set -u
cd /Users/luopeng/Documents/GitHub/domain-adapt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=phase1/.venv/bin/python
BASE=phase1/results/week12_lora_cpt/50_50_fused
DATA=phase1/data/processed/preference/train_split.jsonl
SWEEP=phase1/results/week16_failmode
LOG=$SWEEP/sweep_run.log
mkdir -p "$SWEEP"

LIMIT=300      # 与 week15 baseline 同口径 (clean Δ)
MAXLEN=2048    # 统一 step 时间 (week15 实测 ~17s/step)
LR=5e-6
EPOCHS=1

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# run specs: "name<space>额外的 train_dpo.py flags" (公共 flags 在下面统一加)
ALL_SPECS=(
  "noise_0.1 --beta 0.3 --noise 0.1"
  "noise_0.3 --beta 0.3 --noise 0.3"
  "noise_0.5 --beta 0.3 --noise 0.5"
  "beta_0.01 --beta 0.01"
  "beta_10 --beta 10"
  "ipo_0.3 --beta 0.3 --loss-type ipo"
)

# 选要跑的 run: 命令行参数; 不给则全 6
WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(noise_0.1 noise_0.3 noise_0.5 beta_0.01 beta_10 ipo_0.3)

want() { local n="$1"; for w in "${WANT[@]}"; do [ "$w" = "$n" ] && return 0; done; return 1; }

log "==== Week16 failure-mode sweep START (base=50_50_fused, limit=$LIMIT, max_len=$MAXLEN, lr=$LR, epochs=$EPOCHS) ===="
log "runs: ${WANT[*]}"

for spec in "${ALL_SPECS[@]}"; do
  name="${spec%% *}"
  extra="${spec#* }"
  if ! want "$name"; then continue; fi
  out=$SWEEP/$name
  if [ -f "$out/adapter_model.safetensors" ]; then
    log "---- SKIP $name (已训完, marker $out/adapter_model.safetensors) ----"
    continue
  fi
  log "---- TRAIN $name [$extra] → $out ----"
  if $PY -u phase1/week15/train_dpo.py \
        --model "$BASE" --data "$DATA" \
        --lr "$LR" --epochs "$EPOCHS" --limit "$LIMIT" --max-length "$MAXLEN" \
        --output "$out" $extra >>"$LOG" 2>&1; then
    log "✓ TRAIN $name OK (adapter @ $out)"
  else
    log "✗ TRAIN $name FAIL (见 $LOG)"
  fi
done

log "==== Week16 failure-mode sweep END (runs done; eval 由 run_dpo_eval.py + eval_winrate.py 接力) ===="
