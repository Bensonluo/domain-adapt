# 推导 2: Softmax + Cross-Entropy 梯度

## 已知

```
z = [z_1, z_2, ..., z_K]              logits 向量
p_i = exp(z_i) / Σ_j exp(z_j)         softmax 概率
L = -log(p_y)                         cross-entropy loss (y 是正确类别)
```

**目标**：求 ∂L/∂z_i

---

## 方法一：直接展开

L = -log(p_y) = -z_y + log(Σ_j exp(z_j))

对 z_i 求偏导，分两种情况：

### 情况 1：i = y

```
∂L/∂z_y = -1 + exp(z_y) / Σ_j exp(z_j)
         = -1 + p_y
         = p_y - 1
```

推导：
- ∂(-z_y)/∂z_y = -1
- ∂(log Σ_j exp(z_j))/∂z_y = exp(z_y) / Σ_j exp(z_j) = p_y

### 情况 2：i ≠ y

```
∂L/∂z_i = 0 + exp(z_i) / Σ_j exp(z_j)
         = p_i
```

推导：
- ∂(-z_y)/∂z_i = 0（z_y 和 z_i 无关）
- ∂(log Σ_j exp(z_j))/∂z_i = exp(z_i) / Σ_j exp(z_j) = p_i

### 合并

```
∂L/∂z_i = p_i - 𝟙(i=y) = p_i - y_i
```

其中 y_i 是 one-hot 标签的第 i 个分量。

**直觉**：预测概率 p 减去目标 one-hot 向量。如果预测正确（p_y ≈ 1），梯度接近 0；如果预测错误，梯度大，推动修正。

---

## 方法二：链式法则（更严谨）

L = -log(p_y)，用链式法则：

```
∂L/∂z_i = ∂L/∂p_y × ∂p_y/∂z_i + Σ_{j≠y} ∂L/∂p_j × ∂p_j/∂z_i
```

但 L 只依赖 p_y，所以 ∂L/∂p_j = 0 (j ≠ y)：

```
∂L/∂z_i = (-1/p_y) × ∂p_y/∂z_i
```

求 ∂p_y/∂z_i（softmax 的 Jacobian）：

### i = y 时

```
∂p_y/∂z_y = ∂/∂z_y [exp(z_y) / Σ exp(z_j)]
          = [exp(z_y) × Σ exp(z_j) - exp(z_y) × exp(z_y)] / (Σ exp(z_j))²
          = p_y(1 - p_y)
```

代入：

```
∂L/∂z_y = (-1/p_y) × p_y(1 - p_y) = -(1 - p_y) = p_y - 1
```

### i ≠ y 时

```
∂p_y/∂z_i = ∂/∂z_i [exp(z_y) / Σ exp(z_j)]
          = [0 × Σ - exp(z_y) × exp(z_i)] / (Σ exp(z_j))²
          = -p_y × p_i
```

代入：

```
∂L/∂z_i = (-1/p_y) × (-p_y × p_i) = p_i
```

### 合并

```
∂L/∂z_i = p_i - 𝟙(i=y)  ✓
```

两种方法结果一致。

---

## 向量化形式

对整个 logits 向量 z：

```
∇_z L = p - y_onehot     (K 维向量)
```

- p：softmax 输出的概率分布
- y_onehot：正确类别的 one-hot 编码

**这意味着**：softmax + cross-entropy 的梯度就是预测误差本身，极其简洁。

---

## 数值稳定性：log-sum-exp trick

直接计算 log(Σ exp(z_j)) 会溢出：

```
log(Σ exp(z_j)) = log(Σ exp(z_j - z_max)) + z_max
```

其中 z_max = max(z)。这样 exp 的参数 ≤ 0，不会溢出。

**PyTorch 实现**：`torch.nn.functional.cross_entropy` 内部用 `log_softmax`，自动做这个优化。

```python
# 等价但数值不稳定的写法
loss = -torch.log(torch.softmax(logits, dim=-1)[labels])

# 数值稳定的写法 (PyTorch 内部)
loss = F.cross_entropy(logits, labels)  # ← 用这个
```

---

## 关键理解

| 问题 | 答案 |
|---|---|
| 梯度的形状是什么？ | 概率分布减去 one-hot，就是"预测误差" |
| 为什么不用 MSE？ | MSE + softmax 的梯度包含 p(1-p) 项，当预测正确时梯度趋近 0（学习慢）；CE 的梯度是 p-1，误差大时梯度大，学习快 |
| 为什么数值稳定很重要？ | float32 最大 ~3.4e38，exp(89) 就溢出了。d_model=4096 的模型 logits 可达数百 |
| 实际训练中谁算这个？ | PyTorch 的 `cross_entropy` 内部融合了 softmax + log + CE，一步出梯度，无需显式算 softmax 概率 |

---

**拍照存档**: `phase0/notes/week7_derivation_softmax_ce.jpg`
