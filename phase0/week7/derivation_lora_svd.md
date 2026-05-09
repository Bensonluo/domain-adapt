# 推导 3: LoRA 的 SVD 视角

## 低秩分解

LoRA 的核心: ΔW = BA, 其中 B ∈ R^{d×r}, A ∈ R^{r×k}

TODO: 证明 rank(ΔW) ≤ r

提示:
- rank(BA) ≤ min(rank(B), rank(A))
- B 的列数 = r, A 的行数 = r
- 所以 rank(BA) ≤ r

## SVD 分解

对预训练权重 W_0 做 SVD:
```
W_0 = U Σ V^T
```

其中:
- U ∈ R^{d×d}, 正交矩阵
- Σ ∈ R^{d×k}, 对角矩阵 (奇异值 σ_1 ≥ σ_2 ≥ ... ≥ 0)
- V ∈ R^{k×k}, 正交矩阵

## 低秩近似

取前 r 个奇异值:
```
W_0 ≈ U_r Σ_r V_r^T
```

其中 U_r ∈ R^{d×r}, Σ_r ∈ R^{r×r}, V_r ∈ R^{k×r}

## LoRA 的直觉

TODO: 用自己的话解释
- [ ] 预训练权重 W_0 的有效信息维度是低秩的
- [ ] 微调时的 "变化量" ΔW 也应该是低秩的
- [ ] 为什么 rank=8 通常够用? (奇异值衰减曲线)

## 奇异值衰减

TODO: 画一个示意图
- x轴: 奇异值序号
- y轴: 奇异值大小
- 标注: 前 8 个奇异值包含了多少信息

---

**拍照存档**: `phase0/notes/week7_derivation_lora_svd.jpg`
