"""
Phase 1 Prep: 数据质量校验

校验所有处理后的数据集质量。

Usage:
    python phase1/prep/validate_data.py --dir data/processed/
"""

import json
import argparse
from pathlib import Path


def validate_cpt_data(path: Path) -> dict:
    """校验 CPT 语料"""
    # TODO: 检查 token 数、格式、空行率
    return {"status": "TODO", "path": str(path)}


def validate_sft_data(path: Path) -> dict:
    """校验 SFT 数据集"""
    # TODO: 检查 messages 格式、角色、长度分布
    return {"status": "TODO", "path": str(path)}


def validate_preference_data(path: Path) -> dict:
    """校验偏好数据集"""
    # TODO: 检查 prompt/chosen/rejected 完整性、长度偏差
    return {"status": "TODO", "path": str(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Processed data directory")
    args = parser.parse_args()

    base = Path(args.dir)
    results = []

    # TODO: 遍历数据子目录，运行对应校验
    for subdir in base.iterdir():
        if subdir.is_dir():
            print(f"Validating {subdir.name}...")

    # TODO: 输出校验报告

    print("TODO: Implement data validation")


if __name__ == "__main__":
    main()
