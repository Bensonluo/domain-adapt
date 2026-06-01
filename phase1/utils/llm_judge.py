"""
Phase 1 Utils: LLM-as-Judge

用 GPT-4o 等大模型做 pairwise 或 pointwise 评估。

Usage:
    python phase1/utils/llm_judge.py \
        --model-a ./results/model_a/ \
        --model-b ./results/model_b/ \
        --questions phase1/data/processed/test_questions.jsonl \
        --judge gpt-4o
"""

import argparse
import json
from pathlib import Path


PAIRWISE_PROMPT = """Compare these two answers for the same question.

Question: {question}

Answer A: {answer_a}
Answer B: {answer_b}

Which answer is better? Rate each 1-5 on:
1. Accuracy
2. Completeness
3. Clarity

Respond in JSON format:
{{"winner": "A" or "B" or "tie", "score_a": {{accuracy: N, completeness: N, clarity: N}}, "score_b": {{accuracy: N, completeness: N, clarity: N}}, "reason": "..."}}"""


def judge_pairwise(question: str, answer_a: str, answer_b: str, judge_model: str = "gpt-4o") -> dict:
    """Pairwise comparison"""
    # TODO: 实现 LLM 调用
    prompt = PAIRWISE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b)
    return {"winner": "TODO", "reason": "Not implemented"}


def judge_pointwise(question: str, answer: str, judge_model: str = "gpt-4o") -> dict:
    """Pointwise scoring"""
    # TODO: 实现单条评分
    return {"score": 0, "reason": "Not implemented"}


def analyze_bias(results: list[dict]) -> dict:
    """分析 judge bias（位置偏差、长度偏差）"""
    # TODO: 分析 A/B 胜率是否接近 50/50
    # TODO: 分析 winner 是否偏向更长的回答
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", help="Model A path")
    parser.add_argument("--model-b", help="Model B path")
    parser.add_argument("--questions", required=True, help="Test questions JSONL")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model")
    parser.add_argument("--mode", default="pairwise", choices=["pairwise", "pointwise"])
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    # TODO: 加载问题
    # TODO: 生成两个模型的回答
    # TODO: 运行 judge
    # TODO: 分析 bias
    # TODO: 保存结果

    print("TODO: Implement LLM-as-Judge evaluation")


if __name__ == "__main__":
    main()
