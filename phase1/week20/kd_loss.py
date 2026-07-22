"""
Phase1 Week20 Part A: KD loss (logit-level knowledge distillation)

L = α · CE(student, gold) + (1−α) · T² · KL_restricted(teacher_topk ‖ student_topk)

- CE: 标准 next-token (labels=-100 被 ignore)
- KL restricted: teacher/student 分布都在 teacher top-K tokens 上 renormalize;
  KL(t‖s) = Σ_k t_k (log t_k − log s_k), per position, completion-only (mask)
- 温度 T: 两边 logits/T 后 softmax, 梯度补偿 T² (Hinton KD)
- α ∈ [0,1]: CE 权重 (α=1 → 纯 hard CE = week19 distill; α=0 → 纯 soft KL)

★ fp32 算 softmax/KL (bf16 数值不稳), 这是 logit KD 常见坑.

输入 (调用方负责 causal shift — 即传 logits[:, :-1], labels[:, 1:], teacher_*[:, :-1]):
  student_logits:    (B, L, V)   float
  labels:            (B, L)      long, completion token id / -100 (prompt+pad)
  teacher_tokens:    (B, L, K)   long, teacher top-K token ids (prompt 位填 0, mask 屏蔽)
  teacher_logits_topk:(B, L, K)  float, teacher top-K raw logits (prompt 位填 0)
"""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, labels, teacher_tokens, teacher_logits_topk,
            alpha=0.5, temperature=2.0, ignore_index=-100):
    """
    返回 (total, ce, kl) — 拆开便于训练时分别 log.
    """
    V = student_logits.size(-1)

    # ★ 单份 fp32 副本 (CE + gather 复用). 原写法 .float() 各调一次 → 造两份全 vocab
    # (B,L,151936) fp32 副本(~389MB/份, batch4), MPS 变长 batch 下 caching allocator
    # 碎片化, pool 单调涨到 88GB, step106 OOM. 复用一份 = 回到 week19 CE 量级.
    sl_f = student_logits.float()

    # ① CE on gold (fp32)
    ce = F.cross_entropy(
        sl_f.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )

    # ② student 在 teacher top-K token 上的 logits: gather (复用同一份 fp32)
    student_topk = torch.gather(
        sl_f, dim=-1, index=teacher_tokens
    )  # (B, L, K)

    T = temperature
    # restricted log-softmax over K (renormalize 到 top-K support)
    s = F.log_softmax(student_topk / T, dim=-1)                  # (B, L, K)
    t = F.log_softmax(teacher_logits_topk.float() / T, dim=-1)   # (B, L, K)

    # KL(t ‖ s) per position = Σ_k t_k (log t_k − log s_k)
    kl_per_pos = (t.exp() * (t - s)).sum(dim=-1)                  # (B, L)

    # 只 completion 位置 (labels != -100) 平均
    mask = (labels != ignore_index).to(kl_per_pos.dtype)         # (B, L)
    denom = mask.sum().clamp_min(1.0)
    kl = (kl_per_pos * mask).sum() / denom

    total = alpha * ce + (1.0 - alpha) * (T * T) * kl
    return total, ce, kl


if __name__ == "__main__":
    # —— 单测: 已知分布手验 (kl_loss 正确性的 5 条不变量) ——
    torch.manual_seed(0)
    B, L, V, K = 2, 4, 10, 5
    sl = torch.randn(B, L, V)
    labels = torch.tensor([[3, 5, -100, -100], [7, 2, -100, -100]])
    tk_vals, tk_idx = sl.topk(K, dim=-1)  # teacher == student baseline

    # 1. teacher == student → KL ≈ 0
    _, _, kl = kd_loss(sl, labels, tk_idx, tk_vals, alpha=0.5, temperature=2.0)
    print(f"[1] teacher==student: KL={kl.item():.2e} (应≈0)")
    assert kl.item() < 1e-4, f"KL 应≈0, got {kl.item()}"

    # 2. teacher ≠ student → KL > 0
    td = torch.randn(B, L, V) * 5
    tv2, ti2 = td.topk(K, dim=-1)
    _, _, kl2 = kd_loss(sl, labels, ti2, tv2, alpha=0.5, temperature=2.0)
    print(f"[2] teacher≠student: KL={kl2.item():.4f} (应>0)")
    assert kl2.item() > 0.01

    # 3. α=1 → total == CE (退化为 week19 hard-label SFT)
    lt, _, _ = kd_loss(sl, labels, ti2, tv2, alpha=1.0, temperature=2.0)
    ce_only = F.cross_entropy(sl.float().reshape(-1, V), labels.reshape(-1), ignore_index=-100)
    print(f"[3] α=1: total={lt.item():.4f} == CE={ce_only.item():.4f}")
    assert abs(lt.item() - ce_only.item()) < 1e-5

    # 4. α=0 → total == T²·KL
    lt0, _, kl0 = kd_loss(sl, labels, ti2, tv2, alpha=0.0, temperature=2.0)
    print(f"[4] α=0: total={lt0.item():.4f} == T²·KL={4*kl0.item():.4f}")
    assert abs(lt0.item() - 4 * kl0.item()) < 1e-6

    # 5. prompt 位 (-100) 不贡献 KL (mask 生效): 给 prompt 位 teacher 极端值, KL 不变
    tv3 = tk_vals.clone()
    tv3[:, 2:, :] = 100.0  # prompt 位 (2,3) 极端 → 若 mask 漏会爆炸
    _, _, kl3 = kd_loss(sl, labels, tk_idx, tv3, alpha=0.5, temperature=2.0)
    print(f"[5] prompt 位填极端: KL={kl3.item():.2e} (应≈ test1 的 0)")
    assert kl3.item() < 1e-3

    print("\n✓ kd_loss 5 条不变量全过")
