"""
Week 1 Day 1-2: PyTorch autograd 底层
====================================

目标: 理解 autograd 计算图,手写 2 层 MLP 的 forward + manual backward,
对比 loss.backward() 自动计算的梯度,确认完全一致。

验收点:
    [ ] x = torch.tensor([1.0], requires_grad=True) 之后,x.grad 什么时候有值?
    [ ] loss.backward() 后,每个参数的 .grad 是怎么算出来的?
    [ ] 为什么 x.grad 默认会累加?何时需要 zero_grad()?
    [ ] retain_graph=True 是干嘛的?第二次 backward() 报什么错?
    [ ] 把 with torch.no_grad(): 包住 forward,grad_fn 还在吗?
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Part 1: 最小热身 — 一行代码看 autograd
# ---------------------------------------------------------------------------
def warmup_one_liner() -> None:
    """从最小例子理解: y = x^2 在 x=3 的导数应该是 2x = 6"""
    x = torch.tensor(3.0, requires_grad=True)
    y = x**2
    y.backward()
    print(f"[warmup] x={x.item()}, y={y.item()}, dy/dx={x.grad.item()} (expect 6.0)")
    assert math.isclose(x.grad.item(), 6.0)


# -------------------------------------------------------------------- -------
# Part 2: 2 层 MLP — 自动 vs 手动梯度对比
# ---------------------------------------------------------------------------
#   网络: x ─W1─> h_pre ─ReLU─> h ─W2─> y_pred
#         loss = mean((y_pred - y)^2)
#
#   维度: x (B, D_in) | W1 (D_in, D_h) | h (B, D_h) | W2 (D_h, D_out) | y (B, D_out)
# ---------------------------------------------------------------------------


def init_params(seed: int = 42) -> tuple[torch.Tensor, ...]:
    """固定随机种子,保证 manual / auto 跑出完全相同的初值"""
    torch.manual_seed(seed)
    D_in, D_h, D_out, B = 4, 8, 2, 3

    x = torch.randn(B, D_in)
    y = torch.randn(B, D_out)

    W1 = torch.randn(D_in, D_h, requires_grad=True)
    b1 = torch.zeros(D_h, requires_grad=True)
    W2 = torch.randn(D_h, D_out, requires_grad=True)
    b2 = torch.zeros(D_out, requires_grad=True)

    return x, y, W1, b1, W2, b2


def forward_auto(
    x: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """用 PyTorch autograd 跑 forward,返回 loss (会自动建计算图)"""
    h_pre = x @ W1 + b1
    h = F.relu(h_pre)
    y_pred = h @ W2 + b2
    loss = ((y_pred - y) ** 2).mean()
    return loss


def backward_manual(
    x: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    手动算梯度。按链式法则一步步推
    """
    with torch.no_grad():  # 手动算梯度时不需要计算图
        h_pre = x @ W1 + b1
        h = torch.relu(h_pre)
        y_pred = h @ W2 + b2
        diff = y_pred - y
        B, D_out = diff.shape
        norm = B * D_out

        # 算 d_loss / d_diff
        d_diff = 2 * diff / norm

        # 算 d_W2, d_b2 (用 d_diff 和 h)
        d_W2 = h.T @ d_diff
        d_b2 = d_diff.sum(0)

        # 算 d_h, 然后过 ReLU 的导数得到 d_h_pre
        d_h = d_diff @ W2.T
        d_h_pre = d_h * (h_pre > 0) # ReLU 的导数: 1 if x>0 else 0

        # 算 d_W1, d_b1
        d_W1 = x.T @ d_h_pre
        d_b1 = d_h_pre.sum(0)

        return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


def backward_auto(
    loss: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """用 autograd 算梯度,作为 ground truth"""
    loss.backward()
    return {name: p.grad.clone() for name, p in params.items()}


def compare_gradients() -> None:
    """对比手动和自动梯度,允许 1e-6 误差(浮点 round-off)"""
    x, y, W1, b1, W2, b2 = init_params()

    # 自动版
    loss_auto = forward_auto(x, y, W1, b1, W2, b2)
    grads_auto = backward_auto(loss_auto, {"W1": W1, "b1": b1, "W2": W2, "b2": b2})

    # 清空梯度,准备跑手动版
    for p in (W1, b1, W2, b2):
        p.grad = None

    # 手动版
    grads_manual = backward_manual(x, y, W1, b1, W2, b2)

    print("\n[gradient diff per parameter]")
    for name in ("W1", "b1", "W2", "b2"):
        diff = (grads_auto[name] - grads_manual[name]).abs().max().item()
        ok = "✅" if diff < 1e-6 else "❌"
        print(f"  {ok} {name}: max abs diff = {diff:.2e}")


# ---------------------------------------------------------------------------
# Part 3: 行为差异小实验
# ---------------------------------------------------------------------------
def quirks() -> None:
    """探索 requires_grad / grad_fn / retain_graph / no_grad 的行为"""

    # 实验 1: 只有叶子节点(用户创建的) requires_grad=True 才有 .grad
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = a + 1                # b 是中间节点
    c = (b * b).sum()
    c.backward()
    print(f"\n[quirk1] a.grad={a.grad}  b.grad={b.grad}  (中间节点默认不存 .grad)")

    # 实验 2: grad_fn 存在的条件
    x = torch.tensor([1.0], requires_grad=True)
    y_with = x * 2
    print(f"[quirk2] y.grad_fn = {y_with.grad_fn}")
    with torch.no_grad():
        y_without = x * 2
    print(f"[quirk2] y_without_grad.grad_fn = {y_without.grad_fn} (应该是 None)")

    # 实验 3: retain_graph
    x = torch.tensor([1.0], requires_grad=True)
    y = x ** 2
    y.backward()             # 第一次 OK
    try:
        y.backward()         # 第二次报错: 计算图已被释放
    except RuntimeError as e:
        print(f"[quirk3] 二次 backward 报错(预期): {str(e)[:80]}...")

    # 实验 4: 梯度累加 vs zero_grad
    p = torch.tensor([1.0], requires_grad=True)
    for _ in range(3):
        loss = (p ** 2).sum()
        loss.backward()
    print(f"[quirk4] 三次 backward 不 zero_grad,p.grad = {p.grad.item()} (期望 6.0=2+2+2)")


if __name__ == "__main__":
    print("=" * 60)
    print("Week 1 Day 1-2: PyTorch autograd 实验")
    print("=" * 60)
    warmup_one_liner()

    try:
        compare_gradients()
    except NotImplementedError as e:
        print("   再跑一次对比。")

    quirks()
