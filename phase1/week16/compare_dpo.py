"""
Phase 1 Week 16: DPO 对比评估

完整对比多个 DPO 模型的评估结果。

Usage:
    python phase1/week16/compare_dpo.py \
        --models results/dpo_0.1 results/dpo_0.3 results/dpo_0.5 \
        --baseline results/cpt_sft/ \
        --output phase1/results/week16_dpo_comparison/
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="DPO model paths")
    parser.add_argument("--baseline", required=True, help="Baseline model path")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    # TODO: 加载各模型的评估结果
    # TODO: 对比 benchmark 分数
    # TODO: 分析生成长度偏差
    # TODO: 训练效率对比
    # TODO: 生成对比报告

    print("TODO: Implement DPO comparison")
    for m in args.models:
        print(f"  Model: {m}")
    print(f"  Baseline: {args.baseline}")


if __name__ == "__main__":
    main()
