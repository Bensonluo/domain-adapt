"""
Phase 1 Week 12: 跑 base + week11-CPT-fused 在 CMMLU (medical_cn + general_cn) 评估
================================================================================

存 scores 到 phase1/results/week12_eval/{base_all,cpt_all}/scores_*.json,
供 eval_cpt.py / eval_forgetting.py 复用 (不重跑评估)。

用法:
    python phase1/week12/run_all_eval.py
    # (想改子集/limit 编辑本文件的 TASKS / LIMIT)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_core import TASK_GROUPS, ROOT, resolve_tasks, run_mlx_evaluate

BASE = "models/Qwen3.5-0.8B-Base-ms"
FUSED = str(ROOT / "phase1" / "results" / "week12_eval" / "week11_cpt_fused")
TASKS = resolve_tasks(["medical_cn", "general_cn"])
LIMIT = 100


def main():
    print(f"tasks ({len(TASKS)}): {TASKS}")
    base = run_mlx_evaluate(BASE, TASKS, ROOT / "phase1/results/week12_eval/base_all", limit=LIMIT)
    fused = run_mlx_evaluate(FUSED, TASKS, ROOT / "phase1/results/week12_eval/cpt_all", limit=LIMIT)

    print("\n=== medical_cn (domain gain = fused - base; 正=提升) ===")
    for t in TASK_GROUPS["medical_cn"]:
        if t in base and t in fused:
            print(f"  {t:40s} {base[t]:.3f} -> {fused[t]:.3f}  gain {fused[t] - base[t]:+.3f}")

    print("\n=== general_cn (forgetting rate = (base-fused)/base; 正=遗忘) ===")
    for t in TASK_GROUPS["general_cn"]:
        if t in base and t in fused:
            rate = (base[t] - fused[t]) / base[t] * 100 if base[t] else 0
            print(f"  {t:40s} {base[t]:.3f} -> {fused[t]:.3f}  rate {rate:+.1f}%")


if __name__ == "__main__":
    main()
