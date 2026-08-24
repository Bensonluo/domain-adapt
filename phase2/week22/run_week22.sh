#!/usr/bin/env bash
set -euo pipefail

WEEK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "$WEEK_DIR/.." && pwd)"
PYTHON="${PHASE2_PYTHON:-$PHASE2_ROOT/../phase1/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 2
fi

"$PYTHON" "$WEEK_DIR/validate_week22.py" --scope code
(
  cd "$PHASE2_ROOT/adaptstack"
  "$PYTHON" scripts/train.py --doctor
  "$PYTHON" scripts/train.py --config configs/medical.yaml --show-plan
  "$PYTHON" scripts/train.py --config configs/medical.yaml --dry-run \
    --output artifacts/week22-smoke
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
    "$PYTHON" -m unittest discover -s tests -v
)
"$PYTHON" "$WEEK_DIR/validate_week22.py" --scope complete
echo "Week 22 complete."
