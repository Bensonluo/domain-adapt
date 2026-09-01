# Week 9：CPT 理论

> 方法学纠错与当前允许表述见 [`CORRECTIONS.md`](CORRECTIONS.md)。本页保留历史学习过程，纠错记录优先于旧结论。

> 目标: 理解 Continual Pre-training 的理论基础，精读 3 篇核心论文。
> 预计时间: 10-14 小时

> **为什么学这周**: CPT 是领域适配的一种重要方法，尤其适合 base 对领域语言/分布覆盖不足的情况；是否需要 CPT、收益是否超过 SFT 或 RAG，必须按任务验证。
>
> **思考锚点**: "为什么 CPT 常使用比从零预训练更保守的学习率？合理范围如何由模型、数据和预算验证？"

---

## 论文精读

### 1. Don't Stop Pretraining (Gururangan 2020)

- 链接: https://arxiv.org/abs/2004.10964
- 核心贡献: 证明领域自适应预训练 (TAPT + DAPT) 在多个 NLP 任务上持续提升
- 关注点:
  - Task-Adaptive Pre-training (TAPT) vs Domain-Adaptive Pre-training (DAPT)
  - 数据量与效果的关系
  - 什么时候 CPT 有用、什么时候没用

### 2. HuatuoGPT-II (2023)

- 链接: https://arxiv.org/abs/2311.06750
- 核心贡献: 医疗领域 CPT 范式 — 用 ChatML 格式做 CPT
- 关注点:
  - 医疗 CPT 数据构建方法
  - CPT + SFT 两阶段流程
  - 评估方法（医疗 benchmark 设计）

### 3. BloombergGPT (2023)

- 链接: https://arxiv.org/abs/2303.17564
- 核心贡献: 金融领域从零训练 + 混合数据策略
- 关注点:
  - 通用语料 vs 领域语料混合比例（48.7% 金融 + 51.3% 通用）
  - 数据质量控制流程
  - 评估体系设计

---

## 关键概念整理

### 灾难性遗忘 (Catastrophic Forgetting)

```
问题: CPT 时模型"忘记"预训练学到的通用知识
量化: 遗忘率 = (MMLU_before - MMLU_after) / MMLU_before

缓解方法:
1. 更保守的学习率与调度 — 具体范围需验证
2. Replay buffer — 通用语料比例需消融
3. EWC (Elastic Weight Consolidation) — 对重要参数加正则
4. LoRA-based CPT — 限制可训练参数，可能降低漂移风险，但仍需实测遗忘
```

### 数据混合比例

```
候选设计（不是预先确定的结果）:
- 纯领域 (100%): 领域曝光最高，可能增加通用退化风险
- 70-30 (领域-通用): 中间候选
- 50-50: 通用 replay 更多，但领域增益和遗忘方向都需实测

需要通过实验找到最优比例（Week 12）
```

---

## 交付物

- [x] [`../notes/week9_dont_stop_pretraining.md`](../notes/week9_dont_stop_pretraining.md) — 论文精读笔记
- [x] [`../notes/week9_huatuoGPT_II.md`](../notes/week9_huatuoGPT_II.md) — 论文精读笔记
- [x] [`../notes/week9_bloombergGPT.md`](../notes/week9_bloombergGPT.md) — 论文精读笔记
- [x] [`../notes/week9_cpt_concepts.md`](../notes/week9_cpt_concepts.md) — CPT 关键概念整理（三篇串联成决策框架）

---

## 自测题

1. **为什么 CPT 常用更保守的学习率？如何验证合理范围？**
2. **TAPT 和 DAPT 的区别是什么？各自适用什么场景？**
3. **画出 CPT 数据混合比例 vs 遗忘率的预期 trade-off 曲线**

> 答案: 1) CPT 从已有能力出发，过大的更新可能破坏原表示，因此通常从更保守的学习率开始；具体范围应通过稳定性、领域 gain 和通用退化的受控 sweep 决定。2) TAPT 聚焦目标任务数据，DAPT 聚焦更广领域语料；适用性取决于数据覆盖和成本。3) 不预设单调曲线，冻结训练预算后实测各比例的 paired delta 与不确定性。

---

## 验收清单

- [x] 3 篇论文精读笔记完成
- [x] 能解释为何 CPT 通常从保守学习率开始，并知道具体范围必须实验确定 → 见 [cpt_concepts 第四节](../notes/week9_cpt_concepts.md)
- [x] 能画出 CPT 数据混合比例 vs 遗忘率的预期 trade-off 曲线 → 见 [cpt_concepts 第三节](../notes/week9_cpt_concepts.md)
- [x] CPT 关键概念整理完成
