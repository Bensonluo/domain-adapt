# 推导 5: AdamW 更新规则

## Adam 更新公式

```
m_t = β1 * m_{t-1} + (1-β1) * g_t          # 一阶矩估计 (动量)
v_t = β2 * v_{t-1} + (1-β2) * g_t²         # 二阶矩估计 (RMS)
m̂_t = m_t / (1-β1^t)                        # 偏差修正
v̂_t = v_t / (1-β2^t)                        # 偏差修正
θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)       # 参数更新
```

## AdamW 的修正

TODO: 写出 AdamW 的更新公式

提示: AdamW **解耦**了 weight decay
```
θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
```

注意: weight decay 直接作用在参数上,而不是 gradients 上。

## Adam vs AdamW

| | Adam | AdamW |
|---|---|---|
| Weight decay | 在梯度里 (L2 正则) | 解耦,直接衰减参数 |
| 公式 | g = g + λθ | θ = θ - lr*(... + λθ) |
| 效果 | 和 adaptive LR 耦合 | 更稳定的正则化 |

## 为什么 LLM 用 AdamW?

TODO: 用自己的话解释
- [ ] 解耦 weight decay 在 adaptive learning rate 下更稳定
- [ ] L2 正则 + adaptive LR 会产生 "effective weight decay 随梯度大小变化" 的问题
- [ ] AdamW 避免了这个问题

## 常用超参

```
β1 = 0.9
β2 = 0.999
ε = 1e-8
λ = 0.01  (weight decay)
```

---

**拍照存档**: `phase0/notes/week7_derivation_adamw.jpg`
