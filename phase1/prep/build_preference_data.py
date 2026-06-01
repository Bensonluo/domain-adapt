"""
Phase 1 Prep: 偏好数据集构建

用 teacher/student 模型生成偏好对，进行质量控制和偏差分析。

Usage:
    python phase1/prep/build_preference_data.py --questions data/raw/questions.jsonl --n 2000
"""

import json
import argparse
from pathlib import Path


def generate_preference_pairs(
    questions: list[str],
    teacher_model: str = "deepseek-v3",
    student_model: str = "qwen2.5-3b-instruct",
    n_pairs: int = 2000,
) -> list[dict]:
    """生成偏好数据对"""
    pairs = []
    for q in questions[:n_pairs]:
        # TODO: 实现 LLM 调用
        # 方法1：大模型 vs 小模型
        # chosen = call_llm(teacher_model, q, temperature=0.3)
        # rejected = call_llm(student_model, q, temperature=0.7)

        # 质量控制：跳过长度差异过大的对
        # if abs(len(chosen) - len(rejected)) / max(len(chosen), 1) > 0.5:
        #     continue

        # pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected})
        pass

    return pairs


def analyze_length_bias(pairs: list[dict]) -> dict:
    """分析 chosen/rejected 长度差分布"""
    # TODO: 实现长度偏差分析
    # - 计算每对 chosen - rejected 长度差
    # - 统计均值、中位数、标准差
    # - 标记长度差 > 50% 的对为"潜在偏差"
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="Questions JSONL file")
    parser.add_argument("--output", default="phase1/data/processed/preference/")
    parser.add_argument("--n", type=int, default=2000, help="Number of pairs to generate")
    parser.add_argument("--teacher", default="deepseek-v3")
    parser.add_argument("--student", default="qwen2.5-3b-instruct")
    args = parser.parse_args()

    # TODO: 读取问题列表
    # TODO: 生成偏好对
    # TODO: 长度偏差分析
    # TODO: 保存偏好数据集
    # TODO: 输出统计报告

    print("TODO: Implement preference data building pipeline")


if __name__ == "__main__":
    main()
