# 推导 1: Self-Attention 反向传播

## 已知 (Forward)

```
Q, K, V ∈ R^{T×d_k}           (省略 batch/head 维度)

S = QK^T / √d_k               (T×T)   attention scores
P = softmax(S, dim=-1)         (T×T)   attention weights
O = P @ V                      (T×d_k) output
```

反向传播时，已知上游梯度 ∂L/∂O ∈ R^{T×d_k}，需求 ∂L/∂Q, ∂L/∂K, ∂L/∂V。

---

## Step 1: ∂L/∂V

O = P @ V，P 在反向传播中视为常数矩阵。

```
∂L/∂V = P^T @ ∂L/∂O          (d_k×T) @ (T×d_k) = (T×d_k)
```

**直觉**：每个 V 的行向量收到来自所有 attention position 的加权梯度，权重恰好是 attention weights P。

---

## Step 2: ∂L/∂P

O = P @ V，对 P 求导：

```
∂L/∂P = ∂L/∂O @ V^T          (T×d_k) @ (d_k×T) = (T×T)
```

但这只是链式法则的上半段，还需要乘 ∂P/∂S（softmax 的 Jacobian）。

---

## Step 3: ∂L/∂S (Softmax Jacobian)

P = softmax(S)，softmax 的 Jacobian 为：

```
∂P_ij/∂S_kl = P_ij (δ_{ik} δ_{jl} - P_il δ_{ik})
```

对第 i 行（固定 i），写成矩阵形式：

```
∂P_i/∂S_i = diag(P_i) - P_i P_i^T            (T×T)
```

其中 P_i 是 P 的第 i 行向量。

因此：

```
∂L/∂S_ij = (∂L/∂P)_ij × P_ij - Σ_l (∂L/∂P)_il × P_il × P_ij
```

合并为矩阵形式：

```
∂L/∂S = P ⊙ (∂L/∂P - (∂L/∂P ⊙ P) @ 1_{T×1} @ 1_{1×T})
```

等价写法（逐行）：

```
G = ∂L/∂P                        (T×T)
D = rowsum(G ⊙ P)                (T×1)
∂L/∂S = P ⊙ (G - D @ 1^T)       (T×T)
```

**关键**：softmax 梯度依赖 P 自身，当 P 趋近 one-hot 时，P ⊙ (1-P) → 0，梯度消失。

---

## Step 4: ∂L/∂Q 和 ∂L/∂K

S = QK^T / √d_k，标量缩放因子 1/√d_k 提出来：

```
∂L/∂Q = (∂L/∂S) @ K / √d_k       (T×T) @ (T×d_k) = (T×d_k)
∂L/∂K = (∂L/∂S)^T @ Q / √d_k     (T×T) @ (T×d_k) = (T×d_k)
```

**推导细节**：

S_ij = Σ_m Q_im K_jm / √d_k

∂S_ij/∂Q_im = K_jm / √d_k

因此 ∂L/∂Q_im = Σ_j (∂L/∂S)_ij × K_jm / √d_k

写成矩阵即 ∂L/∂Q = (∂L/∂S) @ K / √d_k。

同理，∂S_ij/∂K_jm = Q_im / √d_k，所以：

∂L/∂K_jm = Σ_i (∂L/∂S)_ij × Q_im / √d_k = (∂L/∂S)^T @ Q / √d_k。

---

## 完整公式汇总

```
∂L/∂V = P^T @ ∂L/∂O
∂L/∂P = ∂L/∂O @ V^T
∂L/∂S = P ⊙ (∂L/∂P - (rowsum(∂L/∂P ⊙ P)) @ 1^T)
∂L/∂Q = ∂L/∂S @ K / √d_k
∂L/∂K = ∂L/∂S^T @ Q / √d_k
```

---

## 关键理解

### 为什么需要 √d_k 缩放？

Q·K 的元素：S_ij = Σ_m Q_im K_jm，是 d_k 个独立随机变量之和。

假设 Q, K 各元素独立且 E[Q_im] = E[K_jm] = 0, Var[Q_im] = Var[K_jm] = σ²，则：

```
E[S_ij] = 0
Var[S_ij] = d_k × σ²
```

方差随 d_k 线性增长。当 d_k = 64 时，Var[S_ij] = 64σ²，标准差 = 8σ。

softmax 在输入方差大时趋近 one-hot（最大值主导），梯度趋近 0。

除以 √d_k 使 Var[S_ij/√d_k] = σ²，保持梯度稳定。

### 梯度爆炸/消失的根源

- **P → one-hot 时**：diag(P) - PP^T 的对角线元素 P_i(1-P_i) → 0，梯度消失
- **P 接近均匀时**：梯度信号最强，学习效率最高
- **√d_k 缩放**的目的就是让 P 不要太尖锐，保持梯度信号

---

**拍照存档**: `phase0/notes/week7_derivation_attention.jpg`
