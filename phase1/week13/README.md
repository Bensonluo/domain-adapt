# Week 13：DPO + GRPO 理论 + 数学推导

> 目标: 深度理解 DPO/GRPO 的数学原理，精读 5 篇论文，手推 DPO loss。
> 预计时间: 14-20 小时

> **上周回顾**: Week 12 你跑完了 CPT 的数据混合 ablation，找到了最优比例。CPT 解决的是"让模型理解领域语言"的问题。这周开始解决"让模型的输出符合人类偏好"的问题。
>
> **思考锚点**: "DPO 为什么能绕过 reward model？GRPO 为什么不需要 reference model？它们各自省掉了什么？"

---

## Day 1-2: DPO 深度精读

### 论文
- **DPO** (Rafailov 2023) — https://arxiv.org/abs/2305.18290
- **IPO** (Azar 2023) — https://arxiv.org/abs/2310.12036 （背景了解，30 分钟）
- **KTO** (Ethayarajh 2024) — https://arxiv.org/abs/2402.01306 （背景了解，30 分钟）

### 关注点
- DPO 如何从 RLHF 目标推导出闭式解
- Bradley-Terry 模型的作用
- Reference model 的作用（KL 约束）
- Beta 参数如何控制对齐强度
- IPO/KTO 与 DPO 的核心区别（不需要成对偏好数据）

---

## Day 3-4: GRPO 深度精读

### 论文
- **DeepSeek-R1** 技术报告 — https://arxiv.org/abs/2501.12948
- **GRPO** (Shao et al. 2024) — https://arxiv.org/abs/2402.03300

### 关注点
- GRPO 的 group-level reward normalization
- 与 PPO/DPO 的本质差异：on-policy vs off-policy
- Reward shaping 和 reward hacking 的识别
- GRPO 在 reasoning tasks 上的优势

---

## Day 5: 数学推导

### 做什么
1. 手推 DPO loss（完整版，从 RLHF 目标开始）
2. 整理 GRPO 机制（与 DPO 的 3 个本质差异）
3. 整理 DPO failure mode（长度偏差、偏好噪声敏感）

### 交付物

- [ ] `derivation_dpo_detailed.md` — DPO loss 完整推导（从 RLHF → Bradley-Terry → 闭式解）
- [ ] `derivation_grpo.md` — GRPO 机制整理 + 与 DPO 对比
- [ ] `notes/week13_dpo_grpo_comparison.md` — 5 篇论文笔记 + 方法对比

---

## 自测题

1. **DPO 的闭式解是怎么消掉 partition function Z(x) 的？**
2. **GRPO 和 DPO 在是否需要 reference model 上的差异是什么？为什么？**
3. **Reward hacking 是什么？在 GRPO 中怎么检测？**

> 答案: 1) 通过 Bradley-Terry 模型，将偏好概率表示为 reward 差的 sigmoid，然后用 reward 函数的反解 r(x,y) = β log(π/π_ref) 代入，Z(x) 在分子分母中抵消。2) DPO 需要 reference model 做 KL 约束（防止策略偏离太远）；GRPO 用 group-level reward normalization 代替绝对 reward，在 group 内相对排序，不需要 reference model。3) Reward hacking = reward 分数上升但实际质量下降。检测方法：定期人工抽样评分，如果 reward 上升但人工分下降 → hacking。

---

## 验收清单

- [ ] 5 篇论文精读完成（DPO/GRPO 必读，IPO/KTO 背景）
- [ ] DPO loss 手推完成
- [ ] 能对比 GRPO vs DPO 的 3 个本质差异
- [ ] 能解释 reward hacking + 检测方法
