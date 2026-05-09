# LoRA 论文精读笔记

> 论文: LoRA: Low-Rank Adaptation of Large Language Models (2021)
> 链接: https://arxiv.org/abs/2106.09685

---

## 核心思想

预训练语言模型的权重矩阵是低秩的,微调时只需要学习一个低秩的增量 ΔW。

公式:
```
W_new = W_0 + (α / r) * B @ A
```
其中:
- W_0 ∈ R^{d×k}: 预训练权重 (冻结)
- B ∈ R^{d×r}: LoRA 矩阵 B
- A ∈ R^{r×k}: LoRA 矩阵 A
- r: rank (通常 8)
- α: scaling 参数 (通常 16)

---

## 手写推导

TODO: 在纸上完成以下推导,拍照存档

1. [ ] 写出 ΔW = BA 的维度变换: R^{r×k} → R^{d×r} → R^{d×k}
2. [ ] 证明 rank(ΔW) ≤ r
3. [ ] 理解 scaling = α / r 的作用: 控制 LoRA 更新的强度

---

## SVD 视角

TODO: 用自己的话解释
- [ ] 预训练权重 W_0 的 SVD 分解: W_0 = UΣV^T
- [ ] 取前 r 个奇异值: W_0 ≈ U_r Σ_r V_r^T
- [ ] 为什么 rank=8 通常够用? (奇异值衰减)

---

## 论文 Table 4: 只适配 q/v 的 ablation

TODO: 记录实验结论
- [ ] 只改 q_proj: 效果?
- [ ] 只改 v_proj: 效果?
- [ ] 改 q+v: 效果?
- [ ] 改所有 projection: 效果? 参数量增加多少?

---

## 初始化策略

TODO: 为什么这样初始化?
- [ ] A 用随机高斯初始化 (small std)
- [ ] B 用 0 初始化
- [ ] 好处: 训练开始时 ΔW = 0,模型表现和预训练一致

---

## 和 Adapter/Prefix-Tuning 的对比

| 方法 | 引入的参数量 | 推理延迟 | 存储 |
|------|-------------|----------|------|
| Full Fine-tuning | 100% | 无 | 每任务一个完整模型 |
| Adapter | ? | ? | ? |
| Prefix-Tuning | ? | ? | ? |
| LoRA | ? | ? | ? |

TODO: 填完上表
