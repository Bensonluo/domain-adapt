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

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_from_scratch import inject_lora, count_trainable_params


def train_with_handwritten_lora(model, input_ids, labels, rank=8, alpha=16, steps=100):
    """用手写的 LoRALinear 训练"""
    inject_lora(model, target_modules=("c_attn",), rank=rank, alpha=alpha)
    print("\n--- 手写 LoRA ---")
    count_trainable_params(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    losses = []
    model.train()

    for step in range(steps):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"[手写] step {step:3d}: loss = {loss.item():.4f}")

    print(f"[手写] step {steps-1:3d}: loss = {loss.item():.4f}")
    return losses


def train_with_peft_lora(model_name, input_ids, labels, rank=8, alpha=16, steps=100):
    """用 PEFT 库的 LoRA 训练（重新加载一个干净模型）"""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("PEFT 未安装,跳过对比")
        return None

    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print("\n--- PEFT LoRA ---")
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    model.train()

    for step in range(steps):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"[PEFT] step {step:3d}: loss = {loss.item():.4f}")

    print(f"[PEFT] step {steps-1:3d}: loss = {loss.item():.4f}")
    return losses


def main():
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # 用随机 token 做 toy 数据（和 lora_from_scratch.py 一致，避免 tokenizer 问题）
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (2, 32))
    labels = input_ids.clone()

    print("=" * 60)
    print("手写 LoRA vs PEFT LoRA 对比")
    print("=" * 60)

    # 1) 手写 LoRA
    model_hand = AutoModelForCausalLM.from_pretrained(model_name)
    losses_hand = train_with_handwritten_lora(model_hand, input_ids, labels, rank=8, alpha=16, steps=100)

    # 2) PEFT LoRA（加载新模型，保证起点一样）
    losses_peft = train_with_peft_lora(model_name, input_ids, labels, rank=8, alpha=16, steps=100)

    # 3) 对比
    if losses_peft is not None:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Loss curve
            axes[0].plot(losses_hand, label="Hand-written LoRA", alpha=0.8)
            axes[0].plot(losses_peft, label="PEFT LoRA", alpha=0.8)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss Curve Comparison")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Loss diff
            diff = [h - p for h, p in zip(losses_hand, losses_peft)]
            axes[1].plot(diff, color="red", alpha=0.8)
            axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Loss Diff (Hand-written - PEFT)")
            axes[1].set_title("Loss Difference")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(os.path.dirname(__file__), "compare_lora.png")
            plt.savefig(save_path, dpi=150)
            print(f"\n图表已保存: week4/compare_lora.png")
            plt.show()
        except ImportError:
            pass

        print(f"\n最终 loss: 手写={losses_hand[-1]:.4f}, PEFT={losses_peft[-1]:.4f}")
        print(f"差值: {abs(losses_hand[-1] - losses_peft[-1]):.4f}")
    else:
        print("\nPEFT 未安装,只完成了手写 LoRA 训练")


if __name__ == "__main__":
    main()
