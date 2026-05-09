"""
Week 2: 领域数据预处理
===================

将原始文本转换为 nanoGPT 训练所需的二进制 token 文件。

用法:
    python phase0/week2/data_prep.py --input data/raw/medical.txt --output data/processed/medical.bin
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[2]


def prepare_data(input_path: str, output_path: str | None = None) -> None:
    """
    读取文本,用 tiktoken (GPT-2 tokenizer) 编码,保存为 uint16 二进制。
    生成 train.bin 和 val.bin (90/10 split)。
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    text = input_file.read_text(encoding="utf-8")
    print(f"原始文本: {len(text):,} 字符")

    # 用 GPT-2 tokenizer (tiktoken)
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    print(f"Token 数: {len(tokens):,}")

    # 90/10 split
    n = int(0.9 * len(tokens))
    train_ids = np.array(tokens[:n], dtype=np.uint16)
    val_ids = np.array(tokens[n:], dtype=np.uint16)

    # 输出路径
    if output_path is None:
        out_dir = ROOT / "phase0" / "data" / "processed"
    else:
        out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(output_path).stem if output_path else input_file.stem
    train_file = out_dir / f"{prefix}_train.bin"
    val_file = out_dir / f"{prefix}_val.bin"

    train_ids.tofile(train_file)
    val_ids.tofile(val_file)
    print(f"训练集: {train_file} ({len(train_ids):,} tokens)")
    print(f"验证集: {val_file} ({len(val_ids):,} tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始文本文件路径")
    parser.add_argument("--output", default=None, help="输出前缀路径")
    args = parser.parse_args()
    prepare_data(args.input, args.output)
