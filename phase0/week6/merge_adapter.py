"""
Week 6: 合并 LoRA Adapter 到 Base Model
=======================================

将训练好的 adapter 权重合并回 base model,得到完整的可独立推理模型。

用法:
    python phase0/week6/merge_adapter.py \
        --adapter ./domain-sft \
        --output ./domain-sft-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_path: str, output_path: str, base_model: str | None = None):
    """
    加载 base model + adapter,合并,保存完整模型。
    """
    # 如果提供了 base_model,用它;否则从 adapter_config.json 里读
    if base_model is None:
        import json
        from pathlib import Path
        config = json.loads(Path(adapter_path, "adapter_config.json").read_text())
        base_model = config["base_model_name_or_path"]
        print(f"从 adapter config 读取 base model: {base_model}")

    print(f"加载 base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"加载 adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("合并 adapter...")
    model = model.merge_and_unload()

    print(f"保存合并后的模型到: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="adapter 目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--base_model", default=None, help="base model ID (默认从 adapter config 读取)")
    args = parser.parse_args()
    merge(args.adapter, args.output, args.base_model)
