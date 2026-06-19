"""
Phase 1 Week 12: 灾难性遗忘评估 (catastrophic forgetting)
==========================================================

对比 CPT 前后的**通用**任务分数, 量化遗忘率。与 eval_cpt.py (测领域提升) 配对。

forgetting_rate = (baseline - cpt) / baseline:
  - 正值 = 遗忘 (通用能力下降)
  - 负值 = 意外提升 (数据混合里含通用数据时常见, 是好事)

看的是 **delta vs base** (0.8B 绝对分接近 25% 随机基线)。完整 trade-off 评估
(forgetting vs domain gain) 留 week13 真数据重训多配比后。

用法:
    python phase1/week12/eval_forgetting.py \\
        --baseline models/Qwen3.5-0.8B-Base-ms \\
        --finetuned phase1/results/week12_eval/week11_cpt_fused

    # 复用上次结果:
    python phase1/week12/eval_forgetting.py \\
        --baseline phase1/results/week12_eval/base/scores.json \\
        --finetuned phase1/results/week12_eval/cpt_week11/scores.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_core import resolve_scores, resolve_tasks


def compute_forgetting_rate(before: dict, after: dict) -> dict:
    """计算遗忘率: (baseline - cpt) / baseline。正=遗忘, 负=意外提升。"""
    results = {}
    for task in before:
        if task in after:
            rate = (before[task] - after[task]) / before[task] if before[task] > 0 else 0
            results[task] = {
                "before": before[task],
                "after": after[task],
                "forgetting_rate": rate,
                "direction": "forgotten" if rate > 0 else "retained",
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="评估灾难性遗忘 (通用任务)")
    parser.add_argument("--baseline", required=True, help="Base 模型路径 或 已有 scores.json")
    parser.add_argument("--finetuned", required=True, help="CPT fused 模型路径 或 已有 scores.json")
    parser.add_argument("--tasks", nargs="+", default=["general_cn"],
                        help="任务组: general_cn (默认, 非医学通用) / medical_cn / medical_en")
    parser.add_argument("--limit", type=int, default=100, help="每任务样本上限 (0.8B 控时)")
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", help="输出 JSON")
    args = parser.parse_args()

    tasks = resolve_tasks(args.tasks)
    print(f"\n[tasks] {tasks}")

    base_scores = resolve_scores(args.baseline, tasks, "base",
                                 args.limit, args.num_shots, args.batch_size, args.seed)
    cpt_scores = resolve_scores(args.finetuned, tasks, "cpt_week11",
                                args.limit, args.num_shots, args.batch_size, args.seed)

    # 只算该任务组的子集 (scores json 可能含全量子集, 避免跨组串扰)
    base_scores = {t: base_scores[t] for t in tasks if t in base_scores}
    cpt_scores = {t: cpt_scores[t] for t in tasks if t in cpt_scores}

    forgetting = compute_forgetting_rate(base_scores, cpt_scores)

    print("\n" + "=" * 50)
    print("Catastrophic Forgetting Report (中文通用)")
    print("=" * 50)
    for task, r in forgetting.items():
        print(f"  {task:40s} | before {r['before']:.3f} → after {r['after']:.3f} "
              f"| rate {r['forgetting_rate']:+.2%} | {r['direction']}")
    avg = None
    if forgetting:
        avg = sum(r["forgetting_rate"] for r in forgetting.values()) / len(forgetting)
        print(f"  {'— 平均 —':40s} | rate {avg:+.2%}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"task_group": args.tasks, "limit": args.limit,
                       "forgetting": forgetting, "avg_rate": avg},
                      f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
