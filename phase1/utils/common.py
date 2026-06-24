"""
Phase 1 Utils: 共享辅助函数

提供各训练脚本通用的基础设施，避免重复代码。
这些函数是"学习无关"的辅助工具，专注于让实验更可复现。
"""

import json
import os
import random
from pathlib import Path
from typing import Optional


def get_device() -> str:
    """检测可用的计算设备

    Returns:
        "cuda", "mps", 或 "cpu"

    思考：为什么 M3 Max 用 mps 而不是 cuda？
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def set_seed(seed: int = 42):
    """设置随机种子，确保实验可复现

    思考：为什么 LLM 训练需要固定种子？哪些组件受种子影响？
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_training_config(output_dir: str, args, training_args=None, extra: Optional[dict] = None):
    """保存训练配置到 JSON，确保实验可复现

    Args:
        output_dir: 输出目录
        args: argparse.Namespace 或 dict
        training_args: transformers.TrainingArguments 或 TRL config
        extra: 额外的配置信息

    思考：被问"你怎么保证实验可复现？"时，这个函数就是答案的一部分。
    """
    config = {
        "script_args": vars(args) if hasattr(args, "__dict__") else args,
    }
    if training_args is not None:
        config["training_args"] = (
            training_args.to_dict()
            if hasattr(training_args, "to_dict")
            else vars(training_args)
        )
    if extra:
        config.update(extra)

    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def load_jsonl(path: str) -> list[dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    """保存 JSONL 文件"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_lines(path: str) -> int:
    """快速统计文件行数"""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def print_dict_table(data: dict, title: str = ""):
    """美观地打印字典表格

    示例：
        print_dict_table({"loss": 0.5, "acc": 0.9}, title="Metrics")
    """
    if title:
        print(f"\n{'=' * 50}")
        print(title)
        print("=" * 50)
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
    for k, v in data.items():
        print(f"  {str(k):>{max_key_len}s} : {v}")
