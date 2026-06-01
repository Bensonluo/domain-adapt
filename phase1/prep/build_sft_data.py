"""
Phase 1 Prep: SFT 数据集构建

从公开数据集筛选 + 格式统一 + 质量分级。

Usage:
    python phase1/prep/build_sft_data.py --input data/raw/ --output data/processed/sft/
"""

import json
import argparse
from pathlib import Path


def validate_sft_entry(entry: dict) -> tuple[bool, str]:
    """检查单条 SFT 数据质量"""
    messages = entry.get("messages", [])
    if len(messages) < 2:
        return False, "对话轮数 < 2"

    # 检查 assistant 回答长度
    for msg in messages:
        if msg["role"] == "assistant":
            if len(msg["content"]) < 20:
                return False, "回答太短"
            if len(msg["content"]) > 2000:
                return False, "回答太长"

    # 检查 template 格式
    if messages[0]["role"] != "system" and messages[0]["role"] != "user":
        return False, "第一条消息角色不对"

    return True, "OK"


def grade_quality(entry: dict) -> str:
    """质量分级：A（高质量）/ B（中等）/ C（基础）"""
    messages = entry.get("messages", [])
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    if not assistant_msgs:
        return "C"

    avg_len = sum(len(m["content"]) for m in assistant_msgs) / len(assistant_msgs)

    # TODO: 实现更精细的质量分级逻辑
    # - A 档：回答结构化、有推理过程、术语准确
    # - B 档：回答完整但缺乏深度
    # - C 档：回答基本可用但有瑕疵

    if avg_len > 200 and len(messages) >= 3:
        return "A"
    elif avg_len > 100:
        return "B"
    else:
        return "C"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory with raw data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train/test split ratio")
    args = parser.parse_args()

    # TODO: 读取原始数据
    # TODO: 运行 validate_sft_entry 过滤
    # TODO: 运行 grade_quality 分级
    # TODO: 划分 train/test
    # TODO: 保存分级数据集
    # TODO: 输出统计报告

    print("TODO: Implement SFT data building pipeline")


if __name__ == "__main__":
    main()
