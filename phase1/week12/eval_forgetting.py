"""
Phase 1 Week 12: 灾难性遗忘评估

对比 CPT 前后的通用 benchmark 分数，量化遗忘率。

Usage:
    python phase1/week12/eval_forgetting.py \
        --baseline Qwen/Qwen2.5-3B \
        --finetuned phase1/results/week12_cpt_70_30/
"""

import argparse
import json


def compute_forgetting_rate(before: dict, after: dict) -> dict:
    """计算遗忘率"""
    results = {}
    for task in before:
        if task in after:
            rate = (before[task] - after[task]) / before[task]
            results[task] = {
                "before": before[task],
                "after": after[task],
                "forgetting_rate": rate,
                "direction": "forgotten" if rate > 0 else "retained",
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="评估灾难性遗忘")
    parser.add_argument("--baseline", required=True, help="Base model path or eval JSON")
    parser.add_argument("--finetuned", required=True, help="CPT model path or eval JSON")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"])
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    # 1. [YOUR CODE] 评估基线模型
    # 提示：
    #   - 调用 utils/eval_benchmark.py 的 run_lm_eval()
    #   - 或直接用 subprocess 跑 lm-eval 命令行
    #   - 记录每个任务的分数 {task_name: score}
    #   - 思考：MMLU 的哪些 subset 最能反映"通用知识"？
    baseline_scores = None  # TODO: 跑 benchmark

    # 2. [YOUR CODE] 评估 CPT 模型
    # 提示：用同样的 tasks 评估 CPT 后的模型
    cpt_scores = None  # TODO: 跑 benchmark

    # 3. 计算遗忘率（已实现）
    # 公式：forgetting_rate = (baseline - cpt) / baseline
    # 正值 = 遗忘，负值 = 意外提升
    forgetting = compute_forgetting_rate(baseline_scores, cpt_scores)

    # 4. [YOUR CODE] 输出报告
    # 提示：
    #   - 打印每个任务的 before/after/forgetting_rate
    #   - 计算平均遗忘率
    #   - 思考：遗忘率 > 20% 时你会怎么调整？
    print("\n" + "=" * 50)
    print("Catastrophic Forgetting Report")
    print("=" * 50)
    for task, result in forgetting.items():
        print(f"  {task:20s} | before: {result['before']:.4f} | after: {result['after']:.4f} | rate: {result['forgetting_rate']:+.2%} | {result['direction']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(forgetting, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
