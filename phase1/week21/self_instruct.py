"""
Phase 1 Week 21: Self-Instruct Pipeline

种子指令 → 大模型生成新问题 → 大模型回答 → 过滤。

Usage:
    python phase1/week21/self_instruct.py \
        --seeds phase1/data/raw/seed_instructions.jsonl \
        --model gpt-4o \
        --n 5000 \
        --output phase1/data/processed/synthetic/
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, help="Seed instructions JSONL")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # TODO: 加载种子指令
    # TODO: 循环生成新指令：
    #   1. 随机选 3-5 条种子作为 few-shot examples
    #   2. 让 LLM 生成新指令
    #   3. 过滤：与已有指令的相似度 < 阈值
    #   4. 让 LLM 回答新指令
    #   5. 质量检查
    # TODO: 保存合成数据
    # TODO: 输出统计报告

    print("TODO: Implement self-instruct pipeline")


if __name__ == "__main__":
    main()
