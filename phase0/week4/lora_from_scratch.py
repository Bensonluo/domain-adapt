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
        # TODO: 实现 forward
        # original = self.original_linear(x)
        # lora_update = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        # return original + lora_update
        raise NotImplementedError("实现 LoRALinear.forward")


def inject_lora(model, target_modules=("c_attn",), rank=8, alpha=16):
    """
    遍历 model 的所有 module,把 target_modules 对应的 Linear/Conv1D 替换成 LoRALinear。
    GPT-2 的 attention 用 c_attn (Conv1D),不是 q_proj/v_proj。
    """
    # TODO: 实现注入逻辑
    # 提示:
    #   for name, module in model.named_modules():
    #       if any(target in name for target in target_modules):
    #           if isinstance(module, (nn.Linear, Conv1D)):
    #               parent_name = ...  # 用 name.rsplit('.', 1) 拆分
    #               setattr(parent, child_name, LoRALinear(module, rank, alpha))
    raise NotImplementedError("实现 inject_lora")


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

    # TODO: 调用 inject_lora
    # inject_lora(model, target_modules=("c_attn",), rank=8, alpha=16)

    print("\n--- 注入 LoRA 后 ---")
    count_trainable_params(model)

    # TODO: 跑一个 toy 训练循环,观察 loss 下降
    # (用一段短文本,训练 100 步即可)


if __name__ == "__main__":
    main()
