"""
Phase 1 Week 12: CPT 模型领域能力评估

评估 CPT 后的模型在领域 benchmark 上的提升，与 eval_forgetting.py（测遗忘）配对使用。

Usage:
    python phase1/week12/eval_cpt.py \
        --baseline Qwen/Qwen2.5-3B \
        --cpt-model phase1/results/week12_cpt_70_30/ \
        --tasks medical

核心学习目标：
1. 理解"领域提升"和"通用遗忘"是 trade-off 关系
2. 学会用同一套 benchmark 对比多个模型
3. 画出"遗忘率 vs 领域提升"的 trade-off 曲线
"""

import argparse
import json


def compute_domain_gain(baseline: dict, cpt_model: dict) -> dict:
    """计算领域提升（与 compute_forgetting_rate 互补）

    Args:
        baseline: 基座模型在领域 benchmark 上的分数 {task_name: score}
        cpt_model: CPT 后模型在同一 benchmark 上的分数 {task_name: score}

    Returns:
        每个任务的提升幅度，包含 absolute_gain / relative_gain / direction
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
    parser = argparse.ArgumentParser(description="评估 CPT 领域提升")
    parser.add_argument("--baseline", required=True, help="Base model path or eval result JSON")
    parser.add_argument("--cpt-model", required=True, help="CPT model path or eval result JSON")
    parser.add_argument("--tasks", nargs="+", default=["medical"],
                        help="Task group: medical, mmlu, all")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    # 1. [YOUR CODE] 加载或评估基线模型
    # 提示：
    #   - 如果传入的是模型路径，调用 utils/eval_benchmark.py
    #   - 如果传入的是 JSON 路径，直接加载已有结果
    #   - 记录每个任务的分数
    baseline_scores = None  # TODO: 评估基线模型

    # 2. [YOUR CODE] 加载或评估 CPT 模型
    # 提示：用同样的 tasks 评估 CPT 后的模型
    cpt_scores = None  # TODO: 评估 CPT 模型

    # 3. 计算领域提升（已实现）
    # absolute_gain = cpt - baseline（正值 = 提升）
    gains = compute_domain_gain(baseline_scores, cpt_scores)

    # 4. [YOUR CODE] 输出报告
    # 提示：
    #   - 打印每个任务的 baseline/after_cpt/gain
    #   - 与 eval_forgetting.py 的结果配对，画出 trade-off 图
    #   - 思考：纯领域 CPT 的领域提升最大，但遗忘也最大——最优甜点在哪？
    print("\n" + "=" * 50)
    print("CPT Domain Gain Report")
    print("=" * 50)
    for task, result in gains.items():
        print(f"  {task:20s} | baseline: {result['baseline']:.4f} | after: {result['after_cpt']:.4f} | gain: {result['absolute_gain']:+.4f} ({result['relative_gain']:+.2%}) | {result['direction']}")

    # [YOUR CODE] 加载 eval_forgetting.py 的结果，计算 trade-off
    # forgetting_path = args.cpt_model.replace("cpt", "forgetting") + ".json"
    # 思考：如何用 4 种混合比例的数据画出"遗忘率 vs 领域提升"曲线？

    if args.output:
        with open(args.output, "w") as f:
            json.dump(gains, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
