"""
Week 3: Qwen-1.5B 全量微调
==========================

目标: 理解全量微调的显存消耗(每个参数都有梯度+优化器状态)。

用法 (GPU 服务器):
    python phase0/week3/train_full_ft.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --data /path/to/domain_data.jsonl \
        --output_dir ./results_full_ft \
        --epochs 3 \
        --batch_size 2 \
        --lr 5e-5

显存估算:
    1.5B 参数 × (模型权重 2B + 梯度 2B + Adam 状态 4B) ≈ 12-15GB
    实际运行时加上激活值,可能需要 20GB+
"""

import argparse
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_data(path: str, tokenizer, max_length: int = 512):
    """加载 JSONL 格式数据,自动 tokenize"""
    dataset = Dataset.from_json(path)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_data(args.data, tokenizer, args.max_length)
    # 90/10 split
    dataset = dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=device == "cuda",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  #  causal LM,不是 masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # 记录显存峰值
    if device == "cuda":
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n显存峰值: {max_mem:.2f} GB")
        with open(Path(args.output_dir) / "memory.txt", "w") as f:
            f.write(f"max_memory_gb: {max_mem:.2f}\n")

    print(f"训练耗时: {elapsed / 60:.1f} 分钟")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="./results_full_ft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    train(args)
