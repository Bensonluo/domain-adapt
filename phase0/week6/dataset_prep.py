"""
Week 6: 领域数据准备
==================

数据清洗 pipeline:
1. 加载原始 JSONL
2. 格式校验 (必须是 OpenAI messages 格式)
3. 去重 (exact match + MinHash)
4. 过滤低质量 (长度、内容质量)
5. 划分 train/test (90/10)

用法:
    python phase0/week6/dataset_prep.py \
        --input data/raw/medical_raw.jsonl \
        --output data/processed/domain_sft.jsonl \
        --test_ratio 0.1
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset


def validate_format(example):
    """校验是否为标准 messages 格式"""
    if "messages" not in example:
        return False, "缺少 messages 字段"
    for msg in example["messages"]:
        if "role" not in msg or "content" not in msg:
            return False, "messages 格式错误"
        if msg["role"] not in ("system", "user", "assistant"):
            return False, f"未知 role: {msg['role']}"
    return True, "OK"


def deduplicate(dataset):
    """简单去重: 基于 md5 hash"""
    seen = set()
    unique = []
    for ex in dataset:
        h = hashlib.md5(json.dumps(ex, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def filter_quality(example):
    """质量过滤"""
    messages = example.get("messages", [])
    # 总长度检查
    total_len = sum(len(m.get("content", "")) for m in messages)
    if total_len < 20:
        return False
    # assistant 回复长度检查
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return False
    for msg in assistant_msgs:
        if len(msg.get("content", "")) < 10:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件 (train)")
    parser.add_argument("--test_output", default=None, help="测试集输出 (默认 output 同目录 test_ 前缀)")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    test_output = Path(args.test_output) if args.test_output else output_path.parent / f"test_{output_path.name}"

    # 加载
    with open(input_path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    print(f"原始数据: {len(raw)} 条")

    # 格式校验
    valid = []
    for ex in raw:
        ok, reason = validate_format(ex)
        if ok:
            valid.append(ex)
        # else: print(f"跳过: {reason}")
    print(f"格式校验通过: {len(valid)} 条")

    # 去重
    unique = deduplicate(valid)
    print(f"去重后: {len(unique)} 条 (去重 {len(valid) - len(unique)} 条)")

    # 质量过滤
    filtered = [ex for ex in unique if filter_quality(ex)]
    print(f"质量过滤后: {len(filtered)} 条 (过滤 {len(unique) - len(filtered)} 条)")

    # 划分
    dataset = Dataset.from_list(filtered)
    split = dataset.train_test_split(test_size=args.test_ratio)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split["train"].to_json(output_path)
    split["test"].to_json(test_output)
    print(f"\n训练集: {output_path} ({len(split['train'])} 条)")
    print(f"测试集: {test_output} ({len(split['test'])} 条)")


if __name__ == "__main__":
    main()
