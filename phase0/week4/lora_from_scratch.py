"""
Week 4: 手写 minimal LoRA
=========================

不依赖 PEFT 库,自己实现 LoRALayer,注入到 GPT-2 的 attention 层。

用法:
    python phase0/week4/lora_from_scratch.py

核心公式:
    W_new = W_0 + (alpha / rank) * B @ A
    其中 B ∈ R^{d_out × r}, A ∈ R^{r × d_in}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


class LoRALinear(nn.Module):
    """
    手写 LoRA 层。
    forward: original(x) + (alpha/rank) * x @ A.T @ B.T
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # GPT-2 使用 Conv1D 而非 nn.Linear, weight shape 不同:
        # nn.Linear: weight shape = [d_out, d_in]
        # Conv1D:    weight shape = [d_in, d_out]
        if isinstance(original_linear, Conv1D):
            d_in, d_out = original_linear.weight.shape
        else:
            d_out, d_in = original_linear.weight.shape

        # LoRA 矩阵
        # A 用小的随机值初始化, B 用 0 初始化 (保证初始时 LoRA 增量=0)
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        # 冻结原始权重
        original_linear.weight.requires_grad = False
        if original_linear.bias is not None:
            original_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始冻结权重的输出
        original = self.original_linear(x)
        # LoRA 增量: x 先过 A (d_in → r), 再过 B (r → d_out), 乘以 scaling
        # x: (B, T, d_in) @ A.T: (d_in, r) → (B, T, r)
        # (B, T, r) @ B.T: (r, d_out) → (B, T, d_out)
        lora_update = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original + lora_update


def inject_lora(model, target_modules=("c_attn",), rank=8, alpha=16):
    """
    遍历 model 的所有 module,把 target_modules 对应的 Linear/Conv1D 替换成 LoRALinear。
    GPT-2 的 attention 用 c_attn (Conv1D),不是 q_proj/v_proj。
    """
    # 先冻结整个模型（和 PEFT 的做法一致）
    for param in model.parameters():
        param.requires_grad = False

    # 遍历模型的所有模块,找到匹配 target_modules 的层并替换
    # 例如 GPT-2 的 "transformer.h.0.attn.c_attn" 包含 "c_attn",就会被替换
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, (nn.Linear, Conv1D)):
                # "transformer.h.0.attn.c_attn" → parent="transformer.h.0.attn", child="c_attn"
                parts = name.rsplit('.', 1)
                parent_name, child_name = parts[0], parts[1]
                # 拿到父模块,用 LoRALinear 替换原始层
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, LoRALinear(module, rank, alpha))


def count_trainable_params(model):
    """统计可训练参数数量和比例"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total:,}")
    print(f"可训练: {trainable:,} ({100 * trainable / total:.2f}%)")
    return trainable, total


def main():
    model_name = "openai-community/gpt2"  # 124M 参数
    print(f"加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n--- 注入 LoRA 前 ---")
    count_trainable_params(model)

    # 注入 LoRA: 把所有 attention 的 c_attn 替换为 LoRALinear
    inject_lora(model, target_modules=("c_attn",), rank=8, alpha=16)

    print("\n--- 注入 LoRA 后 ---")
    count_trainable_params(model)

    # toy 训练循环: 用随机 token 训练 100 步,观察 loss 下降
    vocab_size = len(tokenizer)
    print(f"vocab_size: {vocab_size}")

    # 手动造输入，跳过 tokenizer 的问题
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (2, 32))
    labels = input_ids.clone()

    # 只训练 LoRA 参数 (A 和 B),冻结的原始权重不参与优化
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    model.train()
    print("\n--- Toy 训练 (100 步) ---")
    for step in range(100):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"step {step:3d}: loss = {loss.item():.4f}")

    print(f"step  99: loss = {loss.item():.4f}")
    print("\nLoRA 训练完成 ✅")


if __name__ == "__main__":
    main()
