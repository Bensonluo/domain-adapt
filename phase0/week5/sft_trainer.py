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
        --eval_data /path/to/domain_dev.jsonl \
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


def mask_assistant_labels(
    input_ids: list[int], tokenizer, expected_turns: int | None = None
) -> list[int]:
    """
    将 prompt 部分的 label 设为 -100,只计算 assistant 回复部分的 loss。

    支持多轮 ChatML/Qwen 对话；监督 assistant 正文和对应的 <|im_end|>，
    mask prompt 与角色前缀。marker 缺失时 fail-closed，避免静默改变训练目标。
    """
    labels = [-100] * len(input_ids)

    assistant_marker = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    turn_end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    def find(pattern: list[int], start: int) -> int:
        for index in range(start, len(input_ids) - len(pattern) + 1):
            if input_ids[index : index + len(pattern)] == pattern:
                return index
        return -1

    cursor = 0
    turns_found = 0
    while True:
        marker_start = find(assistant_marker, cursor)
        if marker_start < 0:
            break
        content_start = marker_start + len(assistant_marker)
        end_start = find(turn_end_marker, content_start)
        if end_start < 0:
            raise ValueError(
                "assistant turn 缺少 <|im_end|>；请检查截断长度或 chat template"
            )
        supervised_end = end_start + len(turn_end_marker)
        labels[content_start:supervised_end] = input_ids[content_start:supervised_end]
        turns_found += 1
        cursor = supervised_end

    if turns_found == 0:
        raise ValueError("没有找到 assistant marker；拒绝退化为全序列 loss")
    if expected_turns is not None and turns_found != expected_turns:
        raise ValueError(
            f"assistant turn 数量不一致: messages={expected_turns}, "
            f"tokenized={turns_found}；样本可能被截断"
        )

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
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=512,
                padding=False,
            )
            input_ids = tokenized["input_ids"]
            expected_turns = sum(
                message.get("role") == "assistant" for message in messages
            )
            labels = mask_assistant_labels(input_ids, tokenizer, expected_turns)
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
    train_dataset = load_and_format_data(args.data, tokenizer)
    eval_dataset = load_and_format_data(args.eval_data, tokenizer)

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
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=device == "cuda",
        seed=args.seed,
        data_seed=args.seed,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"训练完成,adapter 保存在: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--eval_data",
        required=True,
        help="预先按 prompt/实体/来源分组得到的独立 dev JSONL；不在训练脚本内随机行切分",
    )
    parser.add_argument("--output_dir", default="./sft_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
