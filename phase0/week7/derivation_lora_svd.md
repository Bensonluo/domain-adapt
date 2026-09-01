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

对任意待分析矩阵 M ∈ R^{d×k} 做 SVD：

```
M = U Σ V^T = Σ_{i=1}^{min(d,k)} σ_i u_i v_i^T
```

- U ∈ R^{d×d}，正交矩阵，列向量 u_i 是左奇异向量
- Σ ∈ R^{d×k}，对角矩阵，σ_1 ≥ σ_2 ≥ ... ≥ σ_r ≥ ... ≥ 0
- V ∈ R^{k×k}，正交矩阵，列向量 v_i 是右奇异向量

### Eckart-Young 定理

在 Frobenius 范数下的最优 rank-r 近似：

```
M^(r) = Σ_{i=1}^{r} σ_i u_i v_i^T = U_r Σ_r V_r^T
```

近似误差：

```
||M - M^(r)||_F² = Σ_{i=r+1}^{min(d,k)} σ_i²
```

### 能量捕获比

前 r 个奇异值捕获的"能量"：

```
E(r) = Σ_{i=1}^{r} σ_i² / Σ_{i=1}^{min(d,k)} σ_i²
```

如果 E(8) > 0.9，只能说明该矩阵在 Frobenius 能量意义下可被 rank-8 较好近似；是否“够用”还必须由下游任务指标验证。对 LoRA 假设，应优先分析未受低秩约束的 `ΔW_full = W_after - W_before`，而不是 W_0 或 LoRA 自身的 BA。

---

## LoRA 为什么有效

### 论文的实验发现

Aghajanyan et al. (2020) 的 intrinsic dimensionality 研究与 LoRA 论文提供了经验动机：

> 在一些任务和模型上，微调优化可以在远低于参数总维度的子空间内取得较好结果。

这不等于以下命题在所有模型和任务上成立：W_0 必然低秩、ΔW 必然快速谱衰减、或 r=4/8 必然接近 full fine-tuning。rank 仍是需要受控验证的容量超参数。

### 数学解释

微调目标：W = W_0 + ΔW，其中 W_0 已编码了大部分语言知识。

ΔW 往往相对 W_0 是任务特定修正，但“幅度小”不推出“秩低”；低秩性是要验证的结构假设：

```
ΔW ≈ B @ A (rank-r)
用受限的 r 维更新子空间表达任务适配
```

### 当前实证状态

Week 4 `lora.ipynb` 使用随机 256×256 矩阵演示 SVD，不是 Qwen2.5-3B 的 q_proj，也不是训练后的真实 ΔW。此前记录的 rank 8/16/32 能量数字没有对应证据，现已撤回。

要关闭本项，应对 full fine-tuning 的真实 `ΔW_full` 使用平方奇异值能量，覆盖多个层和模块，并结合相同条件的 rank 4/8/16/32 下游消融。不能分析 LoRA 的 BA 来“证明”低秩，因为 BA 的秩在参数化时已经被限制。

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

**作用**：把更新容量 r 与显式缩放 alpha 分开表达，方便组织超参搜索；它不保证改变 rank 后无需重新调 learning rate。

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

可把 α 直观理解为缩放、r 理解为容量，但初始化、scaling、优化器和 learning rate 会共同影响更新动态。改变 r 后仍需验证 alpha 与 learning rate 的组合。

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

该矩阵示例中可训练参数减少约 128 倍；效果是否接近 full fine-tuning 取决于模型、任务、数据和训练设置，不能由参数量公式推出。

---

## 关键理解

| 问题 | 答案 |
|---|---|
| 为什么考虑低秩更新？ | 预训练权重提供强先验，已有研究观察到部分任务的有效更新具有较低内在维度；这是经验动机，不是普遍定理 |
| rank=8 是否够用？ | 没有通用数学保证；应结合真实 ΔW 谱与受控下游消融判断 |
| alpha 和 rank 各自的角色？ | alpha 控制强度，rank 控制自由度（容量） |
| LoRA 节省多少参数？ | 例：d=k=4096, r=16 时，仅需 0.78% 的参数 |

---

**拍照存档**: `phase0/notes/week7_derivation_lora_svd.jpg`
