"""
Week 5-6: 完整 SFT 训练脚本
===========================

支持:
- Chat Template (apply_chat_template)
- Loss Masking (只计算 assistant response 的 loss)
- QLoRA (4-bit NF4 量化 + LoRA)
- 学习率调度 (warmup + cosine)

用法 (GPU 服务器):
    python phase0/week5/sft_trainer.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data /path/to/domain_data.jsonl \
        --output_dir ./sft_output \
        --epochs 3 \
        --lora_r 16 \
        --lora_alpha 32
"""

from __future__ import annotations

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def mask_assistant_labels(input_ids: list[int], tokenizer) -> list[int]:
    """
    将 prompt 部分的 label 设为 -100,只计算 assistant 回复部分的 loss。

    简化实现: 找 <|im_start|>assistant 之后的位置 (ChatML/Qwen2 格式)。
    更完整的实现见 loss_masking.py。
    """
    labels = [-100] * len(input_ids)

    # 编码 assistant 标记
    try:
        assistant_marker = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
    except Exception:
        return input_ids.copy()

    # 从后向前搜索 assistant_marker 的最后出现位置
    marker_len = len(assistant_marker)
    for i in range(len(input_ids) - marker_len, -1, -1):
        if input_ids[i : i + marker_len] == assistant_marker:
            for j in range(i + marker_len, len(input_ids)):
                labels[j] = input_ids[j]
            break

    # 如果没找到 marker, fallback 到全部计入 loss
    if all(l == -100 for l in labels):
        labels = input_ids.copy()

    return labels


def load_and_format_data(path: str, tokenizer):
    """加载 JSONL 数据,格式化为对话格式,tokenize + loss masking"""
    dataset = Dataset.from_json(path)

    def format_example(examples):
        all_input_ids = []
        all_labels = []
        for messages in examples["messages"]:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = tokenizer(
                full_text, truncation=True, max_length=512, padding=False
            )
            input_ids = tokenized["input_ids"]
            labels = mask_assistant_labels(input_ids, tokenizer)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        return {"input_ids": all_input_ids, "labels": all_labels}

    return dataset.map(
        format_example, batched=True, remove_columns=dataset.column_names
    )


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if device == "cuda":
        # QLoRA: NF4 量化 + 双重量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 数据
    dataset = load_and_format_data(args.data, tokenizer)
    dataset = dataset.train_test_split(test_size=0.1)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=device == "cuda",
        report_to="none",
    )

    # DataCollatorForSeq2Seq 处理动态 padding + labels padding(-100)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"训练完成,adapter 保存在: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default="./sft_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    train(args)
