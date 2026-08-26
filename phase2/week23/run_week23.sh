#!/usr/bin/env bash
set -euo pipefail

WEEK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "$WEEK_DIR/.." && pwd)"
PYTHON="${PHASE2_PYTHON:-$PHASE2_ROOT/../phase1/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 2
fi

"$PYTHON" "$WEEK_DIR/validate_week23.py" --scope code
echo "NOT IMPLEMENTED: Week 23 业务代码和真实产物尚未完成。" >&2
echo "按 $WEEK_DIR/README.md 的 Day 顺序实现后，再启用本脚本中的业务命令。" >&2
exit 3
