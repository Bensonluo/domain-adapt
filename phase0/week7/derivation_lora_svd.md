# 推导 3: LoRA 的 SVD 视角

## 低秩分解

LoRA 的核心假设：微调时的权重变化量 ΔW 可以用低秩矩阵近似。

```
ΔW = B @ A
B ∈ R^{d×r},  A ∈ R^{r×k},  r ≪ min(d, k)
```

### 证明 rank(ΔW) ≤ r

```
rank(BA) ≤ min(rank(B), rank(A))
rank(B) ≤ r  (B 只有 r 列)
rank(A) ≤ r  (A 只有 r 行)
⇒ rank(ΔW) ≤ r
```

**直觉**：ΔW 的列空间 ⊆ B 的列空间，最多 r 维。ΔW 的所有信息都压缩在 r 个自由度里。

---

## SVD 分解与低秩近似

对预训练权重 W_0 ∈ R^{d×k} 做 SVD：

```
W_0 = U Σ V^T = Σ_{i=1}^{min(d,k)} σ_i u_i v_i^T
```

- U ∈ R^{d×d}，正交矩阵，列向量 u_i 是左奇异向量
- Σ ∈ R^{d×k}，对角矩阵，σ_1 ≥ σ_2 ≥ ... ≥ σ_r ≥ ... ≥ 0
- V ∈ R^{k×k}，正交矩阵，列向量 v_i 是右奇异向量

### Eckart-Young 定理

在 Frobenius 范数下的最优 rank-r 近似：

```
W_0^(r) = Σ_{i=1}^{r} σ_i u_i v_i^T = U_r Σ_r V_r^T
```

近似误差：

```
||W_0 - W_0^(r)||_F² = Σ_{i=r+1}^{min(d,k)} σ_i²
```

### 能量捕获比

前 r 个奇异值捕获的"能量"：

```
E(r) = Σ_{i=1}^{r} σ_i² / Σ_{i=1}^{min(d,k)} σ_i²
```

如果 E(8) > 0.9，说明前 8 个奇异值捕获了 90%+ 的信息，rank-8 近似就够了。

---

## LoRA 为什么有效

### 论文的实验发现

Aghajanyan et al. (2020) "Intrinsic Dimensionality" 发现：

> 预训练模型的权重矩阵具有**低内在维度**（low intrinsic dimensionality），即微调时的有效更新方向集中在少数几个维度上。

具体表现：
1. W_0 的奇异值**衰减很快** — 少数方向承载大部分信息
2. ΔW（微调变化量）的奇异值衰减**更快** — 更新量本身也是低秩的
3. 实验中 r=4 或 r=8 就能达到接近 full fine-tuning 的效果

### 数学解释

微调目标：W = W_0 + ΔW，其中 W_0 已编码了大部分语言知识。

ΔW 只需要做"小修正"，修正量自然集中少数方向：

```
W_0 ≈ U_r Σ_r V_r^T + U_{rest} Σ_{rest} V_{rest}^T
             ↑ 主要信息         ↑ 次要信息 (已由 W_0 捕获)

ΔW ≈ B @ A (rank-r)
只需要调整主要方向的系数
```

### 实际数据（来自 Week 4 lora.ipynb）

对 Qwen2.5-3B 的 q_proj 层：
- rank=8 捕获约 85-95% 的 ΔW 能量
- rank=16 捕获约 95-99%
- rank=32 基本接近 100%

**为什么论文用 r=8 就够了**：因为 ΔW 不是 W_0，ΔW 是微调变化量，它的秩比 W_0 更低。

---

## alpha 的作用

LoRA 的实际更新：

```
h = W_0 x + (α/r) × B A x
```

- α（alpha）：LoRA 的"总强度"
- r（rank）：LoRA 的"自由度"
- scaling = α/r

### 为什么不直接设 scaling = 1（alpha = rank）？

**好处**：调 rank 时不需要重新调 learning rate。

举例：
- r=8, α=16 → scaling = 2, effective LR = 2 × lr
- r=16, α=16 → scaling = 1, effective LR = 1 × lr
- r=16, α=32 → scaling = 2, effective LR = 2 × lr

**固定 α=16**：
- r=8 → scaling=2
- r=16 → scaling=1

**固定 α=32**：
- r=8 → scaling=4
- r=16 → scaling=2

用 α 来控制强度，r 来控制容量。调 r 时 α 不变，总强度只下降一半（而不是剧烈变化），learning rate 不需要跟着调。

### 直觉类比

- **alpha** = 音量旋钮（控制 LoRA 更新的总幅度）
- **rank** = 均衡器频段数（控制 LoRA 能调整多少个方向）
- scaling = alpha/rank = 每个频段的平均增益

---

## 与 Full Fine-Tuning 的对比

```
Full FT:  W = W_0 + ΔW           参数量 = d × k
LoRA:     W = W_0 + (α/r) B A    参数量 = (d + k) × r

例: d=k=4096, r=16
  Full FT: 4096 × 4096 = 16.8M 参数
  LoRA:    (4096+4096) × 16 = 131K 参数 (0.78%)
```

训练参数减少 128 倍，但效果接近 full fine-tuning。这就是低秩假设的力量。

---

## 关键理解

| 问题 | 答案 |
|---|---|
| 为什么低秩假设成立？ | 预训练权重已有强先验，微调只需小幅修正，修正量自然低秩 |
| rank=8 够用的数学依据？ | ΔW 的奇异值衰减比 W_0 更快，8 个方向足以表达更新量 |
| alpha 和 rank 各自的角色？ | alpha 控制强度，rank 控制自由度（容量） |
| LoRA 节省多少参数？ | 例：d=k=4096, r=16 时，仅需 0.78% 的参数 |

---

**拍照存档**: `phase0/notes/week7_derivation_lora_svd.jpg`
