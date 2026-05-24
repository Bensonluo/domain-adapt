"""
Week 1 Day 5-7: Transformer Block + 完整训练循环
==================================================

把 Day 3-4 的 attention 拼成一个完整的 mini-GPT,在 toy 数据(tiny_shakespeare)上
跑训练循环,看到 loss 下降。

模块组成:
    LayerNorm  ─┐
                ├─ Block: x + Attention(LN(x))
    FFN        ─┘         x + FFN(LN(x))
    位置编码 + 多个 Block 堆叠 = MiniGPT

视频对照:
- Karpathy "Let's build GPT" 全视频: https://www.youtube.com/watch?v=kCc8FmEb1nY

跑法:
    python phase0/week1/day5_7_transformer.py     # 测试模块
    python phase0/week1/train_toy.py              # 跑训练

验收点:
    [ ] 能在白板上画完整 forward 路径(从 input_ids 到 logits)
    [ ] 理解为什么 Pre-LN (`x + Attn(LN(x))`) 比 Post-LN 更稳定
    [ ] 能解释 token embedding + position embedding 为什么是相加而非拼接
    [ ] 训练 loss 从 ~4.5 (随机基线 = log(vocab_size)) 降到 < 2.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. LayerNorm — 手写不用 nn.LayerNorm
# ---------------------------------------------------------------------------
class MyLayerNorm(nn.Module):
    """
    标准化最后一维: y = gamma * (x - mean) / sqrt(var + eps) + beta
    其中 mean, var 在最后一维上算 (per-token,不是 per-batch)
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实现 LayerNorm,验收时和 F.layer_norm 输出一致
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# ---------------------------------------------------------------------------
# 2. Causal Self-Attention — 给 Block 用 (复用 Day 3-4 的思路,但更紧凑)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    (B, T, D) → (B, T, D),内部走 multi-head + causal mask。
    生产代码会用 F.scaled_dot_product_attention 加速,这里也用它.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)                                  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                     # 各 (B, T, D)
        # 拆 head: (B, T, D) -> (B, H, T, D_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # SDPA + causal
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 合并 head: (B, H, T, D_k) -> (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.proj(out))


# ---------------------------------------------------------------------------
# 3. FFN — Linear → GELU → Linear
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    """标准 Transformer FFN: 升维 4x → 激活 → 降维"""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实现 forward — 升维 → GELU → 降维 → dropout
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# 4. Transformer Block — Pre-LN 残差结构
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """
    Pre-LN 结构 (GPT-2 之后主流):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    对比 Post-LN(原始 Transformer):
        x = LayerNorm(x + Attention(x))
    Pre-LN 在深层网络更稳定(梯度直接走残差)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = MyLayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = MyLayerNorm(d_model)
        self.ffn = FFN(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 5. MiniGPT — 整体模型
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        block_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = MyLayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享: tok_emb 和 head (常见技巧,省参数)
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx     : (B, T) 整数 token ids
        targets : (B, T) 整数 token ids(下一 token),用于计算 loss
        返回    : logits (B, T, V), loss (标量 or None)
        """
        B, T = idx.shape
        assert T <= self.block_size, f"序列长度 {T} 超出 block_size {self.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B, T, D)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """从 idx 开始,自回归生成 max_new_tokens 个新 token"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # 只看最后一位
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 6. 自检
# ---------------------------------------------------------------------------
def smoke_test() -> None:
    """模型结构跑通"""
    torch.manual_seed(0)
    try:
        model = MiniGPT(vocab_size=100, d_model=64, n_heads=4, n_layers=2, block_size=32)
        idx = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))
        logits, loss = model(idx, targets)
        assert logits.shape == (2, 16, 100)
        assert loss.item() > 0
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[smoke test] forward OK,参数量 = {n_params:,},初始 loss = {loss.item():.3f}")
        print(f"             随机基线 loss ≈ log(100) = {math.log(100):.3f},应该接近")
    except NotImplementedError as e:
        print(f"⚠️  你还有 TODO 没填: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 1 Day 5-7: Transformer Block + MiniGPT 模块测试")
    print("=" * 60)
    smoke_test()
    print("\n模块跑通后,执行 train_toy.py 开始训练。")
