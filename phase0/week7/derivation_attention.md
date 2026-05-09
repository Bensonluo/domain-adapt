# 推导 1: Self-Attention 反向传播

## 已知

Forward:
```
S = QK^T / √d_k          (B, H, T, T)
O = softmax(S) @ V       (B, H, T, d_k)
```

## TODO: 推导 ∂L/∂Q

提示:
1. 先求 ∂L/∂S
2. S = QK^T / √d_k
3. 用矩阵求导法则

## TODO: 推导 ∂L/∂K

## TODO: 推导 ∂L/∂V

提示: O = A @ V, 其中 A = softmax(S)
∂L/∂V = A^T @ ∂L/∂O

## 关键理解

为什么 attention 梯度容易爆炸?
- 当 softmax 输出接近 one-hot 时, 梯度趋近于 0 (梯度消失)
- 当 attention score 很大时, softmax 的导数可能很大 (梯度爆炸)
- /√d_k 的缩放就是为了缓解这个问题

---

**拍照存档**: 手写推导照片 → `phase0/notes/week7_derivation_attention.jpg`
