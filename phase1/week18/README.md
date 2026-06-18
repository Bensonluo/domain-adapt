# Week 18：Distillation 理论

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
3. **On-policy distillation 为什么效果最好但最贵？**

> 答案: 1) 训练算法几乎相同（都是交叉熵 SFT），区别在**数据来源与信息量**：普通 SFT 用人工标注的 hard label（1 bit 信息），response distillation 用 teacher 生成的输出（可拿 soft label，携带"类间关系"的 dark knowledge）；成本上，模型造数据可无限放大、远比人工便宜，天花板是 teacher 能力而非标注预算。2) 因为 teacher 和 student 的 hidden state **维度通常不同**（如 768d vs 384d），无法直接算距离；projection layer 是一个可学习的线性映射 `W: student_dim → teacher_dim`，把 student 表示投影到 teacher 空间再对齐，从而**解耦 student 架构**（允许砍宽度）。DistilBERT 不需要 projection，是因为它故意保持 hidden size=768 与 BERT 相同，只砍层数。3) **效果最好**因为 student 在自己的分布上探索（on-policy），能发现 teacher 没示范的好路径，本质是 RL，**能超越 teacher**（R1 的 aha moment）；**最贵**因为：多轮迭代（生成→打分→训练循环）、大量采样（每样本生成 N 条只用相对信号）、需要 reward（规则 reward 限可验证任务，神经 RM 有 hacking 风险）、在线生成（GPU 开销大）。详见 [comparison 第六节](../notes/week18_distillation_comparison.md)。

---

## 验收清单

- [x] 3 篇论文精读完成（DistilBERT / Self-Instruct / Zephyr）
- [x] 三种蒸馏方法对比表完成（Response / Feature / On-Policy + 决策框架）
- [x] 能解释每种方法的适用场景 → 见 [comparison 第四节](../notes/week18_distillation_comparison.md)
