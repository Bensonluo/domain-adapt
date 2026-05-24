"""
Week 1 Day 3-4: 手写 Multi-Head Attention
=========================================

目标: 不依赖 nn.MultiheadAttention,从零实现 scaled dot-product attention,
加上 causal mask 和 multi-head 拆分,然后用 F.scaled_dot_product_attention
作为参考验证一致性。

视频对照:
- Karpathy "Let's build GPT" (前 30 分钟到 attention): https://www.youtube.com/watch?v=kCc8FmEb1nY
- Attention is all you need: https://arxiv.org/abs/1706.03762

跑法:
    python phase0/week1/day3_4_attention.py

验收点:
    [ ] 在白板上画出 (B, T, D) → (B, H, T, D_k) → attention → (B, T, D) 的完整 shape 流
    [ ] 能解释 causal mask 为什么是上三角(对角线以上设 -inf)
    [ ] 能解释 / sqrt(d_k) 缩放的意义(softmax 的方差爆炸)
    [ ] 一致性测试通过 (max abs diff < 1e-5 vs F.scaled_dot_product_attention)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Part 1: 手写实现
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    输入  : x  shape = (B, T, D_model)
    输出  : y  shape = (B, T, D_model)

    内部维度变换:
        x  → W_q/W_k/W_v → q/k/v   shape = (B, T, D_model)
        reshape + transpose        → (B, H, T, D_k)   D_k = D_model / H
        attention(q, k, v)         → (B, H, T, D_k)
        merge heads                → (B, T, D_model)
        W_o                        → (B, T, D_model)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能整除 n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 用 Linear 而不是 Parameter,因为 Linear 自带初始化和 bias
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # ----- 投影 + reshape 到 (B, H, T, D_k) -----

        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # ----- 算 attention scores -----
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)        # (B, H, T, T)

        # ----- 加 causal mask -----
        # mask 形状 (T, T),上三角(diagonal=1)为 True
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # ----- softmax + 加权求和 -----
        attn = F.softmax(scores, dim=-1)
        # out  = attn @ v                     # (B, H, T, D_k)
        out  = self.dropout(attn) @ v

        # ----- 合并 heads + 输出投影 -----
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(self.dropout(out))


# ---------------------------------------------------------------------------
# Part 2: 用 PyTorch 的 SDPA 作为参考实现
# ---------------------------------------------------------------------------
def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    F.scaled_dot_product_attention 是 PyTorch 内置的高效 SDPA,
    用作 ground truth 验证我们的手写实现。

    输入: q, k, v 都是 (B, H, T, D_k)
    输出: (B, H, T, D_k)
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ---------------------------------------------------------------------------
# Part 3: 一致性测试
# ---------------------------------------------------------------------------
def test_consistency() -> None:
    """
    思路:
      - 手动算一次 q/k/v 投影
      - 一份用我们的 attention 公式
      - 一份用 F.scaled_dot_product_attention
      - 对比输出
    """
    torch.manual_seed(0)
    B, T, D, H = 2, 5, 32, 4
    D_k = D // H

    # 随便造点权重和输入
    x = torch.randn(B, T, D)
    W_q = nn.Linear(D, D, bias=False)
    W_k = nn.Linear(D, D, bias=False)
    W_v = nn.Linear(D, D, bias=False)

    # 投影 + reshape (B, T, D) -> (B, H, T, D_k)
    def proj(W: nn.Linear) -> torch.Tensor:
        return W(x).view(B, T, H, D_k).transpose(1, 2)

    q, k, v = proj(W_q), proj(W_k), proj(W_v)

    # 手写版
    scale = 1.0 / math.sqrt(D_k)
    scores = (q @ k.transpose(-2, -1)) * scale            # (B, H, T, T)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out_manual = attn @ v                                 # (B, H, T, D_k)

    # 参考版
    out_ref = reference_attention(q, k, v)

    diff = (out_manual - out_ref).abs().max().item()
    ok = "✅" if diff < 1e-5 else "❌"
    print(f"[consistency test] manual vs SDPA max abs diff = {diff:.2e}  {ok}")


# ---------------------------------------------------------------------------
# Part 4: 测试整个 Module
# ---------------------------------------------------------------------------
def test_module_runs() -> None:
    """跑一遍 forward,验证 shape 正确"""
    torch.manual_seed(0)
    B, T, D, H = 2, 8, 64, 4
    mha = MultiHeadAttention(d_model=D, n_heads=H)
    x = torch.randn(B, T, D)
    try:
        y = mha(x)
        assert y.shape == (B, T, D), f"输出 shape 错误: {y.shape}"
        print(f"[module test] 输入 {tuple(x.shape)} → 输出 {tuple(y.shape)} ✅")
    except NotImplementedError as e:
        print(f"⚠️  你还没填 TODO: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 1 Day 3-4: 手写 Multi-Head Attention")
    print("=" * 60)
    print("\n--- Part 3: 公式一致性 (不依赖你的 Module 实现) ---")
    test_consistency()

    print("\n--- Part 4: 你的 MultiHeadAttention Module 是否跑通 ---")
    test_module_runs()
