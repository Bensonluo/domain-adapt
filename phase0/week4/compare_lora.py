"""
Week 4: 手写 LoRA vs PEFT LoRA 对比实验
========================================

在同样的数据和模型上,分别用:
1. 手写的 LoRALinear (lora_from_scratch.py)
2. PEFT 库的 LoRA (peft.LoraConfig)

对比 loss 下降曲线和参数量。

用法:
    python phase0/week4/compare_lora.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def train_with_peft_lora(model, tokenizer, texts, rank=8, alpha=16, steps=100):
    """用 PEFT 库的 LoRA 训练"""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("PEFT 未安装,跳过对比")
        return None

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for step in range(steps):
        batch = texts[step % len(texts)]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=64)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"[PEFT] step {step}: loss = {loss.item():.4f}")

    return losses


def main():
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 造点 toy 数据
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "PyTorch is an open source machine learning framework.",
    ] * 10

    print("=" * 60)
    print("手写 LoRA vs PEFT LoRA 对比")
    print("=" * 60)

    # TODO: 先实现 lora_from_scratch.py,然后在这里加载手写版模型训练
    # TODO: 再加载 PEFT 版模型训练
    # TODO: 对比两者的 loss 曲线和参数量

    print("\n请先在 lora_from_scratch.py 中完成 TODO,再运行本脚本。")


if __name__ == "__main__":
    main()
