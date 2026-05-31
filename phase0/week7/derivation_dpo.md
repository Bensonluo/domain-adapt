# 推导 4: DPO Loss 推导

## 起点：RLHF 目标

RLHF 优化的是带 KL 约束的奖励最大化：

```
max_π  E_{x~D, y~π(·|x)}[r(x,y)] - β × KL(π(·|x) || π_ref(·|x))
```

- π：当前策略（要训练的模型）
- π_ref：参考策略（SFT 阶段的模型，固定不动）
- r(x,y)：reward model 给出的奖励
- β：KL 惩罚系数，控制策略不要偏离 π_ref 太远

**RLHF 的问题**：需要先训一个 reward model，再用 PPO 等在线算法优化策略。训练不稳定、工程复杂。

**DPO 的目标**：跳过 reward model，直接用偏好数据训练策略。

---

## Step 1：RLHF 目标的最优解

将 RLHF 目标展开：

```
max_π  E[r(x,y)] - β × E[log(π(y|x)/π_ref(y|x))]
```

这是一个带约束的优化问题。对每个 x，在所有 y 上最大化：

```
L(π) = Σ_y π(y|x) [r(x,y) - β log(π(y|x)/π_ref(y|x))]
```

用变分法（对 π(y|x) 求导令其为 0）：

```
∂L/∂π(y|x) = r(x,y) - β [log(π(y|x)/π_ref(y|x)) + 1] = 0

⇒ log(π(y|x)/π_ref(y|x)) = r(x,y)/β - 1

⇒ π(y|x) = π_ref(y|x) × exp(r(x,y)/β) / Z(x)
```

其中归一化常数：

```
Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)
```

这就是最优策略的闭式解。

**关键问题**：Z(x) 需要对所有可能的 y 求和，不可计算。所以 RLHF 用 PPO 近似优化。

---

## Step 2：反解 reward

从 Step 1 的闭式解，反解 reward 函数：

```
π*(y|x) = π_ref(y|x) × exp(r(x,y)/β) / Z(x)

⇒ π*(y|x) / π_ref(y|x) = exp(r(x,y)/β) / Z(x)

⇒ log(π*(y|x) / π_ref(y|x)) = r(x,y)/β - log Z(x)

⇒ r(x,y) = β log(π*(y|x) / π_ref(y|x)) + β log Z(x)
```

**关键观察**：β log Z(x) 只依赖 x，不依赖 y。

---

## Step 3：Bradley-Terry 偏好模型

人类偏好建模为 Bradley-Terry 模型。给定一对回答 (y_w, y_l)，其中 y_w 是人类偏好的（winner），y_l 是不偏好的（loser）：

```
P(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))
```

其中 σ 是 sigmoid：σ(z) = 1/(1+exp(-z))。

---

## Step 4：消掉 Z(x)

将 Step 2 的 reward 代入 Step 3：

```
r(x,y_w) - r(x,y_l)
= [β log(π*(y_w|x)/π_ref(y_w|x)) + β log Z(x)]
  - [β log(π*(y_l|x)/π_ref(y_l|x)) + β log Z(x)]
= β [log(π*(y_w|x)/π_ref(y_w|x)) - log(π*(y_l|x)/π_ref(y_l|x))]
```

**Z(x) 完全消掉了！** 因为它在成对比较中相减为零。

代入 Bradley-Terry：

```
P(y_w ≻ y_l | x) = σ(β [log π*(y_w|x)/π_ref(y_w|x) - log π*(y_l|x)/π_ref(y_l|x)])
```

---

## Step 5：构造 DPO Loss

用 π_θ 替换 π*（用当前可训练的策略模型），取 negative log-likelihood：

```
L_DPO(θ) = -E_{(x,y_w,y_l)~D} [log σ(β (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

展开 log-ratio：

```
L_DPO(θ) = -E [log σ(β (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))]
```

用代码表示：

```python
def dpo_loss(log_pi_w, log_pi_l, log_ref_w, log_ref_l, beta):
    log_ratio_w = log_pi_w - log_ref_w     # winner 的 log-ratio
    log_ratio_l = log_pi_l - log_ref_l     # loser 的 log-ratio
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

---

## DPO 的梯度分析

对 log π_θ(y_w|x) 和 log π_θ(y_l|x) 求梯度：

```
∂L/∂log π_θ(y_w|x) = -β × σ(-logits)       → 增大好回答的概率
∂L/∂log π_θ(y_l|x) = +β × σ(-logits)        → 减小坏回答的概率
```

**直觉**：
- 当 y_w 的 log-ratio 远大于 y_l 的（模型已学对）→ σ(-logits) → 0 → 梯度消失（已收敛）
- 当 y_w 和 y_l 差不多（模型没学好）→ σ(-logits) → 0.5 → 梯度最大（学习信号最强）

---

## 完整推导流程图

```
RLHF 目标
  │
  ├─ 求最优解 → π* = π_ref × exp(r/β) / Z(x)
  │                         ↑ Z(x) 不可计算
  │
  ├─ 反解 reward → r(x,y) = β log(π*/π_ref) + β log Z(x)
  │                              ↑ Z(x) 依赖 x，不依赖 y
  │
  ├─ Bradley-Terry → P(y_w > y_l) = σ(r_w - r_l)
  │
  ├─ 代入 reward → r_w - r_l = β [log(π*/π_ref)]_w - β [log(π*/π_ref)]_l
  │                ↑ Z(x) 相减消掉了！
  │
  └─ 取 NLL → L_DPO = -E[log σ(β (log_ratio_w - log_ratio_l))]
```

---

## 关键理解

| 问题 | 答案 |
|---|---|
| DPO 消掉了什么？ | 配分函数 Z(x)，通过成对比较中 Z(x) 相减为零 |
| DPO 相对 RLHF 的优势？ | 不需要 reward model，不需要 PPO，离线训练，工程简单得多 |
| β 的作用？ | 控制策略偏离 π_ref 的程度。β 大 → 更保守，β 小 → 更激进 |
| DPO 的局限？ | 依赖静态偏好数据，无法在线探索；偏好数据质量决定上限 |

---

**参考**:
- DPO 论文: https://arxiv.org/abs/2305.18290
- https://huggingface.co/blog/pref-tuning

**拍照存档**: `phase0/notes/week7_derivation_dpo.jpg`
