"""
Phase 1 Week 12: 评估**真数据** CPT fused(base 复用已有 base_all,只跑真数据 fused)
================================================================================

与 run_all_eval.py 的区别:
- run_all_eval.py: base + week11 假数据 fused 都重跑
- run_real_eval.py: base 复用 base_all/scores_*.json(已存在),只评估真数据 fused

产物: phase1/results/week12_eval/real_cpt_all/scores_real_cpt_fused.json
下游: eval_cpt.py / eval_forgetting.py 读 base_all + real_cpt_all 算 domain_gain_real / forgetting_real

用法:
    python phase1/week12/run_real_eval.py
    # 前置: phase1/results/week12_eval/real_cpt_fused/ 已 fuse 完成
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_core import TASK_GROUPS, ROOT, resolve_tasks, run_mlx_evaluate, load_scores

REAL_FUSED = str(ROOT / "phase1" / "results" / "week12_eval" / "real_cpt_fused")
BASE_ALL = ROOT / "phase1" / "results" / "week12_eval" / "base_all"
TASKS = resolve_tasks(["medical_cn", "general_cn"])
LIMIT = 100


def main():
    if not Path(REAL_FUSED).exists():
        raise SystemExit(f"❌ 真数据 fused 模型不存在: {REAL_FUSED}\n先跑 mlx_lm.fuse 合并 week12_real_cpt/adapters")

    print(f"tasks ({len(TASKS)}): {TASKS}")
    # 只跑真数据 fused; base 复用
    real = run_mlx_evaluate(REAL_FUSED, TASKS, ROOT / "phase1/results/week12_eval/real_cpt_all", limit=LIMIT)
    base = load_scores(BASE_ALL)

    print("\n=== medical_cn (domain gain = real - base; 正=提升) ===")
    for t in TASK_GROUPS["medical_cn"]:
        if t in base and t in real:
            print(f"  {t:40s} {base[t]:.3f} -> {real[t]:.3f}  gain {real[t] - base[t]:+.3f}")

    print("\n=== general_cn (forgetting rate = (base-real)/base; 正=遗忘) ===")
    for t in TASK_GROUPS["general_cn"]:
        if t in base and t in real:
            rate = (base[t] - real[t]) / base[t] * 100 if base[t] else 0
            print(f"  {t:40s} {base[t]:.3f} -> {real[t]:.3f}  rate {rate:+.1f}%")


if __name__ == "__main__":
    main()
