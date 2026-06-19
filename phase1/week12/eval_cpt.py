"""
Phase 1 Week 12: CPT 领域提升评估 (domain gain)
================================================

对比 base vs CPT 模型在中文医疗任务上的提升, 与 eval_forgetting.py (测遗忘) 配对使用。

为什么用 mlx_lm.evaluate (不是 lm_eval --model hf): 我们的模型是 MLX 格式,
lm-eval-harness 原生不认 MLX。详见 _eval_core.py。

⚠️ CPT 模型必须是 **已 fuse 的独立模型** (evaluate CLI 没有 --adapter-path, 源码确认)。

用法:
    # 1. 先 fuse (全量微调 adapters → 独立模型, 无损)
    python -m mlx_lm fuse --model models/Qwen3.5-0.8B-Base-ms \\
        --adapter-path phase1/results/week11_cpt_pure/adapters \\
        --save-path phase1/results/week12_eval/week11_cpt_fused

    # 2. 跑领域提升 (base vs fused)
    python phase1/week12/eval_cpt.py \\
        --baseline models/Qwen3.5-0.8B-Base-ms \\
        --cpt-model phase1/results/week12_eval/week11_cpt_fused

    # 复用上次结果 (不重跑评估):
    python phase1/week12/eval_cpt.py \\
        --baseline phase1/results/week12_eval/base/scores.json \\
        --cpt-model phase1/results/week12_eval/cpt_week11/scores.json

核心学习目标:
1. 领域提升 absolute_gain = CPT 后 - CPT 前 (正值 = 学到了领域知识)
2. 0.8B 绝对分接近随机基线 (25%), 看的是 **delta vs base**, 不是绝对值
3. 假数据 CPT (week11) 预期 domain gain 弱 — 只学 form 不学 fact (见 week11 README)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_core import resolve_scores, resolve_tasks


def compute_domain_gain(baseline: dict, cpt_model: dict) -> dict:
    """计算领域提升 (与 compute_forgetting_rate 互补)。

    absolute_gain = cpt - baseline (正值 = 提升)。relative_gain 相对基线百分比。
    """
    results = {}
    for task in baseline:
        if task in cpt_model:
            gain = cpt_model[task] - baseline[task]
            results[task] = {
                "baseline": baseline[task],
                "after_cpt": cpt_model[task],
                "absolute_gain": gain,
                "relative_gain": gain / baseline[task] if baseline[task] > 0 else 0,
                "direction": "improved" if gain > 0 else "regressed",
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="评估 CPT 领域提升 (domain gain)")
    parser.add_argument("--baseline", required=True, help="Base 模型路径 或 已有 scores.json")
    parser.add_argument("--cpt-model", required=True, help="CPT fused 模型路径 或 已有 scores.json")
    parser.add_argument("--tasks", nargs="+", default=["medical_cn"],
                        help="任务组: medical_cn (默认) / general_cn / medical_en 或具体任务名")
    parser.add_argument("--limit", type=int, default=100, help="每任务样本上限 (0.8B 控时)")
    parser.add_argument("--num-shots", type=int, default=0, help="few-shot (CMMLU 默认 0-shot)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123, help="固定 seed 复现")
    parser.add_argument("--output", help="输出 JSON")
    args = parser.parse_args()

    tasks = resolve_tasks(args.tasks)
    print(f"\n[tasks] {tasks}")

    base_scores = resolve_scores(args.baseline, tasks, "base",
                                 args.limit, args.num_shots, args.batch_size, args.seed)
    cpt_scores = resolve_scores(args.cpt_model, tasks, "cpt_week11",
                                args.limit, args.num_shots, args.batch_size, args.seed)

    # 只算该任务组的子集 (scores json 可能含全量子集, 避免跨组串扰)
    base_scores = {t: base_scores[t] for t in tasks if t in base_scores}
    cpt_scores = {t: cpt_scores[t] for t in tasks if t in cpt_scores}

    gains = compute_domain_gain(base_scores, cpt_scores)

    print("\n" + "=" * 50)
    print("CPT Domain Gain Report (中文医疗)")
    print("=" * 50)
    for task, r in gains.items():
        print(f"  {task:40s} | base {r['baseline']:.3f} → cpt {r['after_cpt']:.3f} "
              f"| gain {r['absolute_gain']:+.3f} ({r['relative_gain']:+.2%}) | {r['direction']}")
    avg = None
    if gains:
        avg = sum(r["absolute_gain"] for r in gains.values()) / len(gains)
        print(f"  {'— 平均 —':40s} | gain {avg:+.3f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"task_group": args.tasks, "limit": args.limit,
                       "gains": gains, "avg_gain": avg}, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
