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

- [ ] `notes/week18_distilbert.md` — 论文精读笔记
- [ ] `notes/week18_self_instruct.md` — 论文精读笔记
- [ ] `notes/week18_zephyr.md` — 论文精读笔记
- [ ] `notes/week18_distillation_comparison.md` — 三种方法对比表

---

## 自测题

1. **Response distillation 和普通 SFT 有什么区别？**
2. **Feature distillation 为什么要加 projection layer？**
3. **On-policy distillation 为什么效果最好但最贵？**

---

## 验收清单

- [ ] 3 篇论文精读完成
- [ ] 三种蒸馏方法对比表完成
- [ ] 能解释每种方法的适用场景
