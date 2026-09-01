"""
Week 6: 领域数据准备
==================

数据清洗 pipeline:
1. 加载原始 JSONL
2. 格式校验 (必须是 OpenAI messages 格式)
3. 精确去重 (规范化 JSON 的 MD5；当前未实现 MinHash 近重复检测)
4. 最小长度过滤（仅作启发式，不等价于完整质量评估）
5. 按指定 group field 划分 train/test，避免同 prompt/实体/来源跨集合

用法:
    python phase0/week6/dataset_prep.py \
        --input data/raw/medical_raw.jsonl \
        --output data/processed/domain_sft.jsonl \
        --eval_output data/processed/domain_dev.jsonl \
        --group_field entity_id \
        --eval_ratio 0.1
"""

import argparse
import hashlib
import json
import random
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
        h = hashlib.md5(
            json.dumps(ex, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique


def passes_length_checks(example):
    """执行最低长度启发式；不声称衡量事实正确性、覆盖或任务质量。"""
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


def get_group_value(example, field_path):
    """读取点号分隔的分组字段，例如 metadata.source_id。"""
    value = example
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"样本缺少分组字段 {field_path!r}: {example}")
        value = value[part]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def split_by_group(examples, group_field, eval_ratio, seed):
    """以组为最小单位切分，确保同组样本不会跨 train/dev。"""
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio 必须在 0 和 1 之间")

    groups = {}
    for example in examples:
        group = get_group_value(example, group_field)
        groups.setdefault(group, []).append(example)

    if len(groups) < 2:
        raise ValueError(
            f"分组字段 {group_field!r} 只有 {len(groups)} 个唯一值，无法切分"
        )

    group_ids = list(groups)
    target_eval_size = max(1, round(len(examples) * eval_ratio))

    # 在多个确定性随机排列上比较前缀，选择最接近目标样本数的非空组集合。
    # 这比“一次打乱后累加到阈值”更能抵抗极不均衡组（如 99/1）。
    rng = random.Random(seed)
    smallest_group = min(group_ids, key=lambda group_id: len(groups[group_id]))
    eval_group_ids = {smallest_group}
    best_error = abs(len(groups[smallest_group]) - target_eval_size)
    attempts = min(256, max(32, len(group_ids)))
    for _ in range(attempts):
        candidate_order = group_ids.copy()
        rng.shuffle(candidate_order)
        candidate_groups = set()
        candidate_size = 0
        for group_id in candidate_order[:-1]:
            candidate_groups.add(group_id)
            candidate_size += len(groups[group_id])
            error = abs(candidate_size - target_eval_size)
            if error < best_error:
                eval_group_ids = candidate_groups.copy()
                best_error = error
            if candidate_size >= target_eval_size:
                break

    train = []
    evaluation = []
    for group_id, group_examples in groups.items():
        destination = evaluation if group_id in eval_group_ids else train
        destination.extend(group_examples)
    return train, evaluation, len(groups), len(eval_group_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 JSONL 文件")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件 (train)")
    parser.add_argument(
        "--eval_output",
        "--test_output",
        dest="eval_output",
        default=None,
        help="开发集输出（--test_output 仅作向后兼容；该集合不得当作最终 blind test）",
    )
    parser.add_argument(
        "--eval_ratio", "--test_ratio", dest="eval_ratio", type=float, default=0.1
    )
    parser.add_argument(
        "--group_field",
        default=None,
        help="用于防泄漏分组的字段，支持点号路径，如 entity_id 或 metadata.source_id",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_random_split",
        action="store_true",
        help="确认数据不存在组级依赖后，显式允许随机行切分",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    eval_output = (
        Path(args.eval_output)
        if args.eval_output
        else output_path.parent / f"dev_{output_path.name}"
    )
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval_ratio 必须在 0 和 1 之间")
    if output_path.resolve() == eval_output.resolve():
        raise ValueError("训练集与开发集不能写入同一路径")

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

    # 最低长度启发式；完整质量审计仍需正确性、来源、覆盖、难度与人工抽检。
    filtered = [ex for ex in unique if passes_length_checks(ex)]
    print(f"长度过滤后: {len(filtered)} 条 (过滤 {len(unique) - len(filtered)} 条)")

    # 划分。默认拒绝随机行切分，避免相同 prompt/实体/来源跨集合。
    if args.group_field:
        train_rows, eval_rows, total_groups, eval_groups = split_by_group(
            filtered, args.group_field, args.eval_ratio, args.seed
        )
        print(
            f"按 {args.group_field!r} 分组: {total_groups} groups, "
            f"dev={eval_groups} groups, achieved_ratio={len(eval_rows) / len(filtered):.3f}"
        )
    elif args.allow_random_split:
        split = Dataset.from_list(filtered).train_test_split(
            test_size=args.eval_ratio, seed=args.seed
        )
        train_rows = list(split["train"])
        eval_rows = list(split["test"])
        print("警告: 使用随机行切分；调用者已显式确认不存在组级依赖")
    else:
        raise ValueError(
            "必须提供 --group_field 进行组级切分；若已确认不存在组级依赖，"
            "请显式传入 --allow_random_split"
        )

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_json(output_path)
    Dataset.from_list(eval_rows).to_json(eval_output)
    print(f"\n训练集: {output_path} ({len(train_rows)} 条)")
    print(f"开发集: {eval_output} ({len(eval_rows)} 条)")


if __name__ == "__main__":
    main()
