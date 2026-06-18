# Week 13：偏好对齐 5 篇论文笔记 + 方法对比

> 把 5 篇论文（DPO / IPO / KTO / GRPO / DeepSeek-R1）串成一套**偏好对齐方法的全景图**，
> 并给出"什么场景该用哪个"的决策框架。配套 [../week13/derivation_dpo_detailed.md](../week13/derivation_dpo_detailed.md) 和 [../week13/derivation_grpo.md](../week13/derivation_grpo.md)。
>
> PDF 原文：[../week13/papers/](../week13/papers/)

---

## 一、全景：偏好对齐方法的演化树

```
                   RLHF (Ouyang 2022)
                   = RM 训练 + PPO 在线采样 + critic 网络
                   痛点: 两阶段 / 在线采样慢 / critic 显存翻倍
                          │
          ┌───────────────┼────────────────────────┐
          ▼               ▼                        ▼
     off-policy         off-policy              on-policy
     (用固定偏好对)     (用单点 good/bad)        (在线采样 + reward)
          │               │                        │
        ┌─┴──┐            │                        │
        ▼    ▼            ▼                        ▼
       DPO  IPO          KTO                     GRPO
      (2023)(2023)      (2024)                  (2024)
       │    │            │                  (DeepSeekMath)
       │  修 DPO 过拟合  不要成对数据         砍 critic + group baseline
       │  (加正则)       (prospect theory)        │
       │                                         ▼
       └──────────────────────────────► DeepSeek-R1 (2025)
                                         纯 GRPO + 规则 reward
                                         → reasoning 涌现
```

**两条路线**：
- **off-policy 路线**（DPO/IPO/KTO）：拿现成偏好数据，把 RL 变成监督学习。轻、稳，但探索性弱。
- **on-policy 路线**（GRPO/R1）：在线采样 + reward，保留 RL 探索，适合 reasoning。

---

## 二、5 篇论文核心笔记

### 1. DPO — Rafailov et al. 2023 (arXiv:2305.18290)

**核心贡献**：把 RLHF 的"训练 reward model + PPO"两阶段，**用一个闭式解 + Bradley-Terry 转化成了单个监督 loss**。

**关键公式**：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\Big[\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\Big]$$

**一句话**：语言模型自己的 logprob 比值就是 reward，不需要单独的 reward model；$Z(x)$ 在 BT 偏好模型里一加一减抵消，绕开了配分函数。

**弱点**：objective 无界（logprob ratio 可推到无穷）→ 过拟合；off-policy 无探索。

> 详见 [../week13/derivation_dpo_detailed.md](../week13/derivation_dpo_detailed.md)

---

### 2. IPO — Azar et al. 2023 (arXiv:2310.12036)

**核心贡献**：提出 **ΨPO 框架**（DPO 是 Ψ=logistic 的特例），令 Ψ=identity 得到 IPO，**专门修 DPO 的无界过拟合**。

**关键区别**：DPO 的 loss 用 $\log\sigma(h)$（$h$ 是 reward 差），当 $h\to\infty$ 时 loss 仍能继续降 → 模型无限推大 logprob ratio。IPO 把目标换成带显式正则的形式（如 $h - \frac{1}{2}h^2$），**让 objective 有界**，并显式控制与 reference 的 KL。

**DPO 的痛点（IPO 论文指出）**：
- loss 无界 → 过拟合
- 实际几乎不 enforce KL（不管 β 多大，策略照样远离 $\pi_{\text{ref}}$）
- 依赖 Bradley-Terry 假设（偏好必须是 BT 可建模的）

**IPO 自己的弱点**：正则是 uniform 的——对"明显偏好"和"模糊偏好"一视同仁，前者太保守。

**一句话**：DPO 的"正则加强版"，牺牲一点对齐强度换稳定性，适合小数据/noisy 偏好。

---

### 3. KTO — Ethayarajh et al. 2024 (arXiv:2402.01306)

**核心贡献**：基于 **Kahneman-Tversky 前景理论**（行为经济学的效用模型），提出**不需要成对偏好数据**的对齐方法。

**关键创新**：
- DPO/IPO 需要 $(y_w, y_l)$ **成对**偏好；KTO 只需要**单点二元标签**（这条输出 desirable / undesirable）。
- 建模**损失厌恶**（loss aversion）：人类对损失的敏感度 > 等量收益。
- loss 分两支：desirable 走 gain branch，undesirable 走 penalty branch，**不对称加权**。

**为什么重要**：成对偏好数据**极贵**（要人工对比两个回答的优劣）；二元 good/bad 标注**便宜几个数量级**。KTO 在 1B–30B 上匹配甚至超过 DPO。

**一句话**：用"好/坏"单点标注替代"A 比 B 好"成对标注，数据成本骤降，效果不输 DPO。

---

### 4. GRPO — Shao et al. 2024, DeepSeekMath §4 (arXiv:2402.03300)

**核心贡献**：PPO 的精简版——**砍掉 critic 网络 $V_\psi$，用一组采样的 reward 做组内 z-score 归一化当 advantage**。

**关键公式**：
$$\hat{A}_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}, \qquad \mathcal{J}_{\text{GRPO}} = \text{PPO-clipped}(\hat{A}) - \beta\,\mathbb{D}_{KL}[\pi_\theta\|\pi_{\text{ref}}]$$

**三处相对 PPO 的改动**：
1. 砍 critic，group baseline 估 advantage
2. KL 从 reward 挪到 loss 外层（不污染 advantage）
3. KL 用 Schulman 2020 无偏估计量

**关键澄清**：GRPO **保留 reference model**（KL），砍的是 **value/critic**。常见误解"GRPO 不需要 reference model"是错的。

> 详见 [../week13/derivation_grpo.md](../week13/derivation_grpo.md)

**一句话**：sample 一组、组内排序当 advantage、PPO 更新、KL 拉回 reference——轻量 RL，专为数学推理优化。

---

### 5. DeepSeek-R1 — DeepSeek-AI 2025 (arXiv:2501.12948)

**核心贡献**：证明 **reasoning 能力可以靠纯 RL（GRPO）涌现**，不需要人类标注的推理链。

**两个模型**：
- **R1-Zero**：DeepSeek-V3-Base 直接上 GRPO，reward 用**规则**（数学正确性 + 格式），**无神经 reward model** → 出现 "aha moment"，推理能力自己冒出来。但可读性差、语言混杂。
- **R1**（工程优化版）：cold-start SFT → RL(GRPO) → rejection sampling 造 SFT 数据 → full SFT → final RL。多阶段把 R1-Zero 的能力固化 + 提升可用性。

**为什么能成**：
1. **规则 reward** = 没有 reward hacking 入口（数学对就是对）
2. **GRPO 的探索性** = 模型自己发现长推理路径（DPO 做不到）
3. **rejection sampling 反向蒸馏** = 把 RL 探索到的好推理，固化进 SFT 数据

**一句话**：纯 RL + 可验证 reward → reasoning 涌现；这是 GRPO 路线的最强证据，也是 2025 年 reasoning 模型的范式起点。

---

## 三、方法对比大表

| 维度 | RLHF/PPO | DPO | IPO | KTO | GRPO |
|------|----------|-----|-----|-----|------|
| **采样范式** | on-policy | off-policy | off-policy | off-policy | on-policy |
| **数据形式** | 偏好对 | 成对 $(y_w,y_l)$ | 成对 $(y_w,y_l)$ | **单点** good/bad | prompt + reward |
| **需要 RM?** | ✅ 显式 | ❌ 隐式 | ❌ 隐式 | ❌ 隐式 | ✅ 显式/规则 |
| **需要 critic?** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **reference model** | ✅(KL in reward) | ✅(logprob 比) | ✅ | ✅ | ✅(独立 KL 项) |
| **objective 有界?** | — | ❌ 无界 | ✅ 有界 | ✅ | — |
| **探索性** | 强 | 弱 | 弱 | 弱 | 强 |
| **数据成本** | 高 | 中（成对） | 中（成对） | **低（单点）** | 中（reward 标注） |
| **典型场景** | 通用对齐 | 通用对齐 | 小数据/noisy | 数据稀缺 | reasoning/可验证任务 |

---

## 四、决策框架：什么场景用哪个

```
你的任务是什么？
│
├─ 任务有"可验证的正确答案"（数学/代码/逻辑）
│   └─→ GRPO（规则 reward，无 hacking，能探索）—— R1 路线
│
├─ 任务只能靠人类偏好判断（开放对话、写作风格）
│   │
│   ├─ 有成对偏好数据 (A 比 B 好)
│   │   ├─ 数据量大、干净 ─────→ DPO
│   │   └─ 数据量小 / 有噪声 ──→ IPO（抗过拟合）
│   │
│   └─ 只有好/坏单点标签（非成对）
│       └─→ KTO（专为非成对设计，数据便宜）
│
└─ 要极致对齐质量 + 有算力
    └─→ RLHF/PPO 或 online DPO（探索 + 在线，最贵最强）
```

**实战速记**：
- **默认首选 DPO**（简单、够用、生态成熟）
- **数据 noisy/小** → IPO
- **只有 like/dislike 数据** → KTO
- **数学/代码/reasoning** → GRPO（你的主攻方向）
- **要 SOTA reasoning** → R1 多阶段管线（cold-start + GRPO + rejection sampling）

---

## 五、知识脉络（一句话串联）

1. **RLHF**（2022）开宗明义：用 RM + PPO 对齐，但工程重。
2. **DPO**（2023）证明：RLHF 目标在 BT 假设下有闭式转化，RL 可省 → off-policy 监督学习。
3. **IPO**（2023）补刀：DPO objective 无界会过拟合，加正则。
4. **KTO**（2024）换角：偏好未必成对，用前景理论处理单点标签。
5. **GRPO**（2024）反向：保留 RL 探索，但砍 critic + group baseline，轻量 on-policy。
6. **DeepSeek-R1**（2025）收官：纯 GRPO + 规则 reward → reasoning 涌现，确立 reasoning 模型新范式。

> **你的方向（蒸馏 + GRPO）**：GRPO 是核心，R1 的"rejection sampling 反向蒸馏"把 RL 能力固化进 SFT 数据——这正是**蒸馏 + GRPO 的交汇点**，值得深挖。

---

## 六、自测题（对应 README）

1. **DPO 的闭式解是怎么消掉 $Z(x)$ 的？**
   通过 Bradley-Terry，偏好概率 = reward 差的 sigmoid；把 implicit reward $r=\beta\log(\pi/\pi_{\text{ref}})+\beta\log Z$ 代入 $r(x,y_w)-r(x,y_l)$，两个 $y$ 共享同一个 $x$，$\beta\log Z(x)$ 一加一减抵消。

2. **GRPO 和 DPO 在 reference model 上的差异？为什么？**（**纠正 README 原答案**）
   **两者都有 reference model**。差异在角色：DPO 以 logprob 比值**隐式**进 loss；GRPO 以独立 KL 项**显式**进 loss。GRPO 砍掉的是 **value/critic 网络**（用 group baseline 代替），不是 reference model。原因：DPO 不采样（不需要 critic 估 advantage）；GRPO 在线采样，但用组内统计量代替 learned critic，省一个模型。

3. **Reward hacking 是什么？GRPO 中怎么检测？**
   reward↑ 但实际质量↓（模型钻 reward 漏洞）。检测：holdout 人工抽样（reward↑ 人工分↓ = hacking）、多 RM 交叉验证、监控 KL 是否飙升。根治：用可验证的规则 reward 替代神经 RM（R1 做法）。
