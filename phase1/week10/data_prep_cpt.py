"""
Phase 1 Week 10: CPT 数据准备脚本

Tokenization + 混合比例配置 + 训练格式切分。

Usage:
    python phase1/week10/data_prep_cpt.py \
        --corpus phase1/data/processed/cpt/ \
        --tokenizer Qwen/Qwen2.5-3B
"""

import argparse
from pathlib import Path


def count_tokens(texts: list[str], tokenizer) -> int:
    """统计 token 总数"""
    total = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        total += len(tokens)
    return total


def mix_data(domain_texts: list[str], general_texts: list[str], ratio: str) -> list[str]:
    """按比例混合领域和通用语料"""
    # ratio: "100-0", "70-30", "50-50"
    domain_pct, general_pct = map(int, ratio.split("-"))
    total = domain_pct + general_pct

    domain_n = int(len(domain_texts) * domain_pct / total)
    general_n = int(len(general_texts) * general_pct / total)

    mixed = domain_texts[:domain_n] + general_texts[:general_n]
    return mixed


def main():
    parser = argparse.ArgumentParser(description="CPT 数据准备")
    parser.add_argument("--corpus", required=True, help="Processed corpus directory")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--ratios", default="100-0,70-30,50-50,30-70", help="Mix ratios (domain-general)")
    parser.add_argument("--seq-length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--output", default="phase1/data/processed/cpt_ready/")
    args = parser.parse_args()

    import os
    os.makedirs(args.output, exist_ok=True)

    # 1. [YOUR CODE] 加载 tokenizer
    # 提示：用 AutoTokenizer.from_pretrained(args.tokenizer)
    # 思考：为什么 CPT 要用目标模型的 tokenizer？不同 tokenizer 的 token 数差异有多大？
    tokenizer = None  # TODO: 加载 tokenizer

    # 2. [YOUR CODE] 读取领域和通用语料
    # 提示：
    #   - 从 args.corpus 目录读取所有文本文件
    #   - 区分 domain_texts 和 general_texts（可用子目录命名）
    #   - 思考：通用语料从哪里来？Wikipedia 中文子集 / Wudao / SkyPile？
    domain_texts = None  # TODO: 读取领域语料
    general_texts = None  # TODO: 读取通用语料

    # 3. [YOUR CODE] 对每种比例生成混合数据
    # 提示：
    #   - 用 mix_data() 按比例混合（已实现）
    #   - 用 tokenizer.encode() 编码
    #   - 按 args.seq_length 切分为固定长度序列（带 overlap 可选）
    #   - 思考：overlap 有什么用？设置多少合适？
    ratios = args.ratios.split(",")
    for ratio in ratios:
        mixed = mix_data(domain_texts, general_texts, ratio)
        # TODO: tokenize 并切分
        # TODO: 保存到 args.output / cpt_{ratio}.bin
        print(f"Ratio {ratio}: TODO - tokenize and save")

    # 4. [YOUR CODE] 输出统计报告
    # 提示：
    #   - 每种比例的 token 总数、平均序列长度
    #   - 与 week11/train_cpt.py 的 max_steps 对照，确认数据量足够
    print("\nTODO: Print token statistics for each ratio")


if __name__ == "__main__":
    main()
