#!/usr/bin/env bash
# Week 21 resumable runner with content-addressed stage reuse.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       MTL_TIMEOUT=0 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

PY=${PY:-phase1/.venv/bin/python}
MLX_PY=${MLX_PY:-/Users/luopeng/Documents/GitHub/4bit-QLoRA-post-training/venv/bin/python}
TEACHER=${TEACHER:-/Users/luopeng/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit}
BASE=${BASE:-phase1/results/week12_lora_cpt/50_50_fused}
CONTROL_MODEL=${CONTROL_MODEL:-phase1/results/week19_distill/real_fused}
SWEEP=${SWEEP:-phase1/results/week21_synthetic}
N_SELF=${N_SELF:-1000}
N_EVOL=${N_EVOL:-250}
DATA=$SWEEP/data
LOG=$SWEEP/run.log
LINEAGE=$SWEEP/artifact_lineage.json
mkdir -p "$DATA"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }
lineage() {
  "$PY" "$SCRIPT_DIR/artifact_lineage.py" "$@" --manifest "$LINEAGE" --sweep "$SWEEP" \
    --base "$BASE" --teacher "$TEACHER" --control-model "$CONTROL_MODEL" \
    --n-self "$N_SELF" --n-evol "$N_EVOL"
}
validate() {
  "$PY" "$SCRIPT_DIR/validate_week21.py" "$@" --sweep "$SWEEP" --reports "$SCRIPT_DIR/results" \
    --base "$BASE" --teacher "$TEACHER" --control-model "$CONTROL_MODEL" --lineage "$LINEAGE" \
    --n-self "$N_SELF" --n-evol "$N_EVOL"
}
archive_stage() {
  local stage=$1
  shift
  local archive=$SWEEP/stale/$(date '+%Y%m%d-%H%M%S')-$stage
  mkdir -p "$archive"
  for target in "$@"; do
    if [ -e "$target" ]; then mv -- "$target" "$archive/"; fi
  done
  log "archived stale $stage artifacts at $archive"
}

if lineage verify --stage seeds >/dev/null 2>&1; then
  log "reuse seed set (source and content lineage verified)"
else
  archive_stage seeds "$DATA/seed_instructions.jsonl"
  log "prepare 100 deterministic seeds (exact holdout exclusion)"
  "$PY" -u "$SCRIPT_DIR/prepare_seeds.py" --output "$DATA/seed_instructions.jsonl" | tee -a "$LOG"
  lineage record --stage seeds
fi

if lineage verify --stage self >/dev/null 2>&1; then
  log "reuse Self-Instruct (content lineage verified)"
else
  if [ -f "$DATA/self_instruct_raw.jsonl" ] && ! lineage verify --stage self --inputs-only >/dev/null 2>&1; then
    archive_stage self "$DATA/self_instruct_raw.jsonl" "$DATA/self_instruct.jsonl" "$DATA/self_instruct_stats.json"
  fi
  log "Self-Instruct: target=$N_SELF"
  "$MLX_PY" -u "$SCRIPT_DIR/self_instruct.py" --seeds "$DATA/seed_instructions.jsonl" \
    --model "$TEACHER" --backend mlx --n "$N_SELF" --batch-size 4 --output "$DATA" --resume | tee -a "$LOG"
  lineage record --stage self
fi

if lineage verify --stage evol >/dev/null 2>&1; then
  log "reuse Evol-Instruct (content lineage verified)"
else
  if [ -f "$DATA/evol_instruct_raw.jsonl" ] && ! lineage verify --stage evol --inputs-only >/dev/null 2>&1; then
    archive_stage evol "$DATA/evol_instruct_raw.jsonl" "$DATA/evol_instruct.jsonl" "$DATA/evol_instruct_stats.json"
  fi
  log "Evol-Instruct: target=$N_EVOL, depth cycles 1..3"
  "$MLX_PY" -u "$SCRIPT_DIR/evol_instruct.py" --input "$DATA/self_instruct.jsonl" \
    --model "$TEACHER" --backend mlx --depth 3 --limit "$N_EVOL" --output "$DATA" --resume | tee -a "$LOG"
  validate --scope data | tee -a "$LOG"
  lineage record --stage evol
fi

log "quality metrics and fixed manual-review sample"
"$PY" -u "$SCRIPT_DIR/assess_quality.py" --self-data "$DATA/self_instruct.jsonl" \
  --evolved-data "$DATA/evol_instruct.jsonl" --real-data phase1/data/processed/cmexam/train.jsonl \
  --output "$SWEEP/quality_metrics.json" --audit-output "$SWEEP/human_audit.jsonl" --audit-size 30 | tee -a "$LOG"
if ! validate --scope audit | tee -a "$LOG"; then
  log "PAUSED: manual review is incomplete. Review $SWEEP/human_audit.jsonl, label it honestly, then rerun."
  exit 2
fi
lineage record --stage quality

log "prepare controlled 50% replacement and full-holdout overlap audit"
"$PY" -u "$SCRIPT_DIR/prepare_replacement.py" \
  --real-sft phase1/results/week19_distill/data/real_sft.jsonl \
  --synthetic "$DATA/evol_instruct.jsonl" "$DATA/self_instruct.jsonl" \
  --holdout phase1/data/processed/cmexam/test.jsonl --output "$DATA/replacement_50.jsonl" \
  --n-total 2000 --synthetic-fraction 0.5 | tee -a "$LOG"
"$PY" -u "$SCRIPT_DIR/check_holdout_similarity.py" --sweep "$SWEEP" \
  --holdout phase1/data/processed/cmexam/test.jsonl --output "$SWEEP/holdout_similarity.json" | tee -a "$LOG"
lineage record --stage replacement

if lineage verify --stage checkpoint >/dev/null 2>&1 && validate --scope checkpoint >/dev/null 2>&1; then
  log "reuse treatment adapter (structure and lineage verified)"
else
  archive_stage checkpoint "$SWEEP/synthetic50" "$SWEEP/synthetic50_fused" "$SWEEP/control_real_fused" "$SWEEP/replacement_statistics.json"
  log "train treatment adapter"
  "$PY" -u phase1/week19/train_distill_sft.py --model "$BASE" --data "$DATA/replacement_50.jsonl" \
    --output "$SWEEP/synthetic50" --lr 2e-5 --epochs 3 --batch-size 4 --grad-accum 4 \
    --max-length 1536 --seed 123 | tee -a "$LOG"
  validate --scope checkpoint | tee -a "$LOG"
  lineage record --stage checkpoint
fi

if lineage verify --stage evaluation >/dev/null 2>&1 && validate --scope evaluation >/dev/null 2>&1; then
  log "reuse fused model and paired evaluation (structure and lineage verified)"
else
  archive_stage evaluation "$SWEEP/synthetic50_fused" "$SWEEP/control_real_fused" "$SWEEP/replacement_statistics.json"
  log "merge treatment adapter"
  "$PY" -u "$SCRIPT_DIR/merge_adapter.py" --base "$BASE" --adapter "$SWEEP/synthetic50" \
    --output "$SWEEP/synthetic50_fused" | tee -a "$LOG"
  log "evaluate all 500 paired holdout predictions for treatment and control"
  "$PY" -u "$SCRIPT_DIR/eval_cmexam_full.py" --model "$SWEEP/synthetic50_fused" \
    --output "$SWEEP/synthetic50_fused/cmexam_holdout.json" --batch-size 8 --max-new-tokens 16 --device cpu | tee -a "$LOG"
  "$PY" -u "$SCRIPT_DIR/eval_cmexam_full.py" --model "$CONTROL_MODEL" \
    --output "$SWEEP/control_real_fused/cmexam_holdout.json" --batch-size 8 --max-new-tokens 16 --device cpu | tee -a "$LOG"
  "$PY" -u "$SCRIPT_DIR/analyze_replacement.py" \
    --control "$SWEEP/control_real_fused/cmexam_holdout.json.preds.jsonl" \
    --treatment "$SWEEP/synthetic50_fused/cmexam_holdout.json.preds.jsonl" \
    --output "$SWEEP/replacement_statistics.json" | tee -a "$LOG"
  validate --scope evaluation | tee -a "$LOG"
  lineage record --stage evaluation
fi

log "write reports and bind final lineage"
"$PY" -u "$SCRIPT_DIR/summarize_week21.py" --sweep "$SWEEP" --reports "$SCRIPT_DIR/results" | tee -a "$LOG"
lineage record --stage summary
validate --scope complete | tee -a "$LOG"
log "Week 21 complete"
