# 推导 2: Softmax + Cross-Entropy 梯度

## 已知

```
z = [z_1, z_2, ..., z_K]          logits
p_i = exp(z_i) / Σ_j exp(z_j)     softmax
L = -log(p_y)                     cross-entropy (y 是正确类别)
```

## TODO: 推导 ∂L/∂z_i

提示:
1. L = -log(p_y) = -z_y + log(Σ_j exp(z_j))
2. 分两种情况: i = y 和 i ≠ y
3. ∂L/∂z_i = p_i - 1(i=y)

## 数值稳定性

log-sum-exp trick:
```
log(Σ exp(z_j)) = log(Σ exp(z_j - z_max)) + z_max
```

为什么? 防止 exp 溢出。

## 验收标准

能在 5 分钟内在白板上独立推完。

---

**拍照存档**: `phase0/notes/week7_derivation_softmax_ce.jpg`
