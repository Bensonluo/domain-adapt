# Week 13：DPO + GRPO 理论 + 数学推导

> 实现边界纠错见 [`CORRECTIONS.md`](CORRECTIONS.md)。理论公式与本项目后续具体框架配置分别说明。

> 目标: 深度理解 DPO/GRPO 的数学原理，精读 5 篇论文，手推 DPO loss。
> 预计时间: 14-20 小时

> **上周回顾**: Week 12 跑完了 CPT 数据混合探索，并选择 50-50 作为后续操作性基线；尚未证明它是最优比例。CPT 旨在调整领域分布建模，这周开始研究偏好对齐。
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

- [x] `derivation_dpo_detailed.md` — DPO loss 完整推导（RLHF → 闭式解 → 反解 reward → BT 抵消 Z(x)）— [derivation_dpo_detailed.md](derivation_dpo_detailed.md)
- [x] `derivation_grpo.md` — GRPO 机制（砍 critic + group baseline）+ 与 DPO 对比 — [derivation_grpo.md](derivation_grpo.md)
- [x] `notes/week13_dpo_grpo_comparison.md` — 5 篇论文笔记 + 方法对比 + 选型决策框架 — [../notes/week13_dpo_grpo_comparison.md](../notes/week13_dpo_grpo_comparison.md)
- [x] 5 篇论文 PDF（对照用）— [papers/](papers/)

---

## 自测题

1. **DPO 的闭式解是怎么消掉 partition function Z(x) 的？**
2. **GRPO 和 DPO 在是否需要 reference model 上的差异是什么？为什么？**
3. **Reward hacking 是什么？在 GRPO 中怎么检测？**

> 答案: 1) 通过 Bradley-Terry 模型，将偏好概率表示为 reward 差的 sigmoid，然后用 implicit reward r(x,y) = β log(π/π_ref) + β log Z(x) 代入 reward 差，两个 y 共享同一个 x，Z(x) 抵消。2) DPO 依赖 reference policy 的 logprob 比值；GRPO 常用 reference KL，但具体实现可在 `beta=0` 时不加载 reference（本项目 Week 17 即如此）。GRPO 的关键简化是用 group baseline 取代 value/critic。3) Reward hacking 是优化代理指标却损害真实目标；可用独立 outcome、盲评和对抗 probes 检测。可验证规则 reward 能降低风险，但不能保证根治所有形式的 hacking。

---

## 验收清单

- [x] 5 篇论文精读完成（DPO/GRPO 必读，IPO/KTO 背景，R1 应用）
- [x] DPO loss 手推完成（4 步：闭式解 → 反解 reward → BT 抵消 Z → loss）
- [x] 能对比 GRPO vs DPO 的 3 个本质差异（采样范式 / reward 来源 / reference 角色）
- [x] 能解释 reward hacking + 检测方法
