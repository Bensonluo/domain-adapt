"""
Phase 1 Week 19: Response Distillation

用大模型回答作为训练数据，训练小模型学习。

Usage:
    python phase1/week19/distill_response.py \
        --questions phase1/data/raw/questions.jsonl \
        --teacher deepseek-v3 \
        --student qwen2.5-3b-instruct \
        --n 5000 \
        --output phase1/data/processed/distill_response/
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="Questions JSONL file")
    parser.add_argument("--teacher", default="deepseek-v3")
    parser.add_argument("--student", default="qwen2.5-3b-instruct")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # TODO: 读取问题列表
    # TODO: 用 teacher 生成回答 (temperature=0.3)
    # TODO: 用 student 生成回答 (temperature=0.7)
    # TODO: 对比 teacher vs student (LLM-as-judge)
    # TODO: 保存蒸馏数据 (teacher 回答作为 SFT 训练数据)
    # TODO: 输出统计报告

    print("TODO: Implement response distillation")


if __name__ == "__main__":
    main()
