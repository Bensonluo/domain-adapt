# Week 18：Distillation 理论

> 方法学纠错见 [`CORRECTIONS.md`](CORRECTIONS.md)。蒸馏、合成数据和 on-policy 的优劣均是条件性问题，不能脱离 teacher、reward、预算和验证成本外推。

> 目标: 理解三种蒸馏方法，精读 3 篇核心论文。
> 预计时间: 10-14 小时

> **思考锚点**: "为什么大模型的回答比小模型好？小模型学大模型的回答能学到什么？学不到什么？"

---

## 论文精读

### 1. DistilBERT (Sanh 2019)
- 链接: https://arxiv.org/abs/1910.01108
- 核心贡献: 蒸馏奠基 — 用 teacher 的 soft label 训练 student
- 关注点: Knowledge distillation loss (KL divergence on logits)、Layer loss

### 2. Self-Instruct (Wang 2022)
- 链接: https://arxiv.org/abs/2212.10560
- 核心贡献: 合成数据生成范式
- 关注点: 种子指令 → 大模型生成新指令 → 过滤 → 训练

### 3. Zephyr 7B (Tunstall 2023)
- 链接: https://arxiv.org/abs/2310.16944
- 核心贡献: 用 GPT-4 蒸馏到 7B 的完整范式
- 关注点: AIF (Alignment Instruction Feedback) 数据 + DPO 对齐

---

## 三种蒸馏方法对比

| 维度 | Response | Feature | On-Policy |
|------|----------|---------|-----------|
| **做法** | 大模型答→小模型学 | 对齐中间层表示 | 小模型生成→大模型评分 |
| **成本** | 低（API 费用） | 中（需访问 hidden） | 高（多轮交互） |
| **深度** | 浅（只学输出） | 中（学内部表示） | 深（学策略） |
| **适用** | 快速验证 | 有 teacher 权重时 | 最终优化 |

---

## 交付物

- [x] [`../notes/week18_distilbert.md`](../notes/week18_distilbert.md) — 论文精读笔记（triple loss + 初始化 ablation + feature distillation 本质）
- [x] [`../notes/week18_self_instruct.md`](../notes/week18_self_instruct.md) — 论文精读笔记（四步自举 + 合成数据 trade-off）
- [x] [`../notes/week18_zephyr.md`](../notes/week18_zephyr.md) — 论文精读笔记（dSFT + AIF + dDPO 三步管线）
- [x] [`../notes/week18_distillation_comparison.md`](../notes/week18_distillation_comparison.md) — 三种方法对比表 + 决策框架 + 自测题解答
- [x] 3 篇论文 PDF（对照用）— [papers/](papers/)

---

## 自测题

1. **Response distillation 和普通 SFT 有什么区别？**
2. **Feature distillation 为什么要加 projection layer？**
3. **On-policy distillation 为什么可能更有效、但通常更贵？**

> 答案: 1) response distillation 与 SFT 常共享交叉熵训练形式，但数据来源和监督信息不同；hard target 不是固定“1 bit”，信息量取决于词表、序列和条件分布。模型生成可扩展，但生成、筛选和事实验证同样有成本。2) teacher/student hidden size 不同时通常需要 projection 或其他对齐方式；架构相同则可能不需要。3) on-policy 方法让 student 在自身分布上采样，可能改善分布匹配，但收益取决于探索、reward、迭代和预算；它不必然最好，也不保证超过 teacher。多轮生成、打分和在线采样通常使其更贵。详见 [comparison 第六节](../notes/week18_distillation_comparison.md)。

---

## 验收清单

- [x] 3 篇论文精读完成（DistilBERT / Self-Instruct / Zephyr）
- [x] 三种蒸馏方法对比表完成（Response / Feature / On-Policy + 决策框架）
- [x] 能解释每种方法的适用场景 → 见 [comparison 第四节](../notes/week18_distillation_comparison.md)
