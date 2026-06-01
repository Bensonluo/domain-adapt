"""
Phase 1 Week 20: On-Policy Distillation

小模型生成回答 → 大模型评分 → 构造偏好对 → DPO。

Usage:
    python phase1/week20/distill_on_policy.py \
        --student phase1/results/week11_cpt_pure/ \
        --judge gpt-4o \
        --output phase1/results/week20_distill_on_policy/
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, help="Student model path")
    parser.add_argument("--data", default="phase1/data/processed/sft/")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model for scoring")
    parser.add_argument("--n-generations", type=int, default=4, help="Generations per prompt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # TODO: 加载 student 模型
    # TODO: 对每个 prompt 生成 n 个回答
    # TODO: 用 judge 对每个回答评分 (1-5)
    # TODO: 构造偏好对 (best vs worst)
    # TODO: 跑 DPO 训练
    # TODO: 评估

    print("TODO: Implement on-policy distillation")


if __name__ == "__main__":
    main()
