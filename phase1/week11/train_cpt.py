"""
Phase 1 Week 11: CPT 训练脚本

在 base model 上做 Continual Pre-training。

Usage:
    python phase1/week11/train_cpt.py \
        --model Qwen/Qwen2.5-3B \
        --data phase1/data/processed/cpt/ \
        --output phase1/results/week11_cpt_pure/

核心学习目标：
1. 理解 CPT 和 SFT 在数据格式上的区别
2. 理解为什么 CPT 学习率比预训练小 10-100 倍
3. 观察 CPT 的 loss 曲线和 SFT 有什么不同
"""

import argparse
import json
import os

# --- [YOUR CODE] 导入必要的库 ---
# 提示：需要 transformers, datasets, torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
# from datasets import load_dataset, load_from_disk
# import torch


# --- 辅助工具（已提供） ---
def get_device() -> str:
    """检测可用的计算设备"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_config(output_dir: str, args, training_args=None):
    """保存训练配置，确保实验可复现"""
    config = {"script_args": vars(args)}
    if training_args is not None:
        config["training_args"] = training_args.to_dict() if hasattr(training_args, "to_dict") else vars(training_args)
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CPT 训练")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Base model name or path")
    parser.add_argument("--data", required=True, help="Training data directory")
    parser.add_argument("--config", help="YAML config file (overrides defaults)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. 设备检测
    device = get_device()
    print(f"Using device: {device}")

    # 2. [YOUR CODE] 加载 base model + tokenizer
    # 提示：
    #   - CPT 要用 base model（非 Instruct 版），如 Qwen/Qwen2.5-3B
    #   - 用 AutoModelForCausalLM.from_pretrained() 加载
    #   - 用 AutoTokenizer.from_pretrained() 加载 tokenizer
    #   - 如果显存不够，考虑用 torch_dtype=torch.bfloat16 或 device_map="auto"
    #   - 思考：为什么 CPT 用 base model 而不是 instruct model？
    model = None  # TODO: AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = None  # TODO: AutoTokenizer.from_pretrained(args.model)

    # 3. [YOUR CODE] 加载并 tokenize 训练数据
    # 提示：
    #   - 数据在 args.data 目录，格式由 week10/data_prep_cpt.py 生成
    #   - 可以用 datasets.load_from_disk() 或自定义 Dataset 类
    #   - 需要把文本转成 input_ids（tokenizer.encode）
    #   - 设置 DataCollatorForLanguageModeling 或自定义 collator
    #   - 思考：CPT 数据不需要 chat template，为什么？
    dataset = None  # TODO: 加载并处理数据
    data_collator = None  # TODO: DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. [YOUR CODE] 配置 TrainingArguments
    # 提示：
    #   - 参考 week11/README.md 的 cpt_config.yaml
    #   - 关键参数：
    #     learning_rate=args.lr（默认 1e-5，比预训练小 10-100 倍）
    #     per_device_train_batch_size=args.batch_size
    #     gradient_accumulation_steps=args.grad_accum
    #     max_steps=args.max_steps
    #     warmup_steps=args.warmup_steps
    #     lr_scheduler_type="cosine"
    #     bf16=True（如果设备支持）
    #   - 思考：为什么 CPT 学习率要比预训练小这么多？
    training_args = None  # TODO: TrainingArguments(output_dir=args.output, ...)

    # 5. [YOUR CODE] 创建 Trainer
    # 提示：
    #   - 用 transformers.Trainer（不是 SFTTrainer）
    #   - model, args, train_dataset, data_collator
    #   - 不需要 eval_dataset（CPT 通常只做训练）
    trainer = None  # TODO: Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)

    # 6. [YOUR CODE] 训练 + 保存
    # 提示：
    #   - trainer.train() 启动训练
    #   - 监控 loss 曲线（可以先用 print，后续加 wandb）
    #   - trainer.save_model() 保存最终模型
    #   - 思考：CPT 的 loss 曲线和 SFT 有什么不同？
    # trainer.train()
    # trainer.save_model(args.output)
    print("TODO: Implement training loop")

    # 7. 保存配置（已提供）
    # save_config(args.output, args, training_args)


if __name__ == "__main__":
    main()
