# Week 9：CPT 理论

> 目标: 理解 Continual Pre-training 的理论基础，精读 3 篇核心论文。
> 预计时间: 10-14 小时

> **为什么学这周**: 垂域大模型的灵魂步骤。不做 CPT = 在不懂医学的脑子上贴便利贴。CPT 让模型从"知道通用知识"变成"理解领域语言"。
>
> **思考锚点**: "为什么 CPT 的学习率要比预训练小 10-100 倍？如果用预训练的学习率会怎样？"

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
1. 学习率调度 — 比预训练小 10-100 倍
2. Replay buffer — 混入 10-30% 通用语料
3. EWC (Elastic Weight Consolidation) — 对重要参数加正则
4. LoRA-based CPT — 只训练少量参数，天然缓解遗忘
```

### 数据混合比例

```
经验值（来自 BloombergGPT + 后续研究）:
- 纯领域 (100%): 最大领域提升，最高遗忘风险
- 70-30 (领域-通用): 平衡方案，遗忘可控
- 50-50: 遗忘最低，但领域提升也最小

需要通过实验找到最优比例（Week 12）
```

---

## 交付物

- [ ] `notes/week9_dont_stop_pretraining.md` — 论文精读笔记
- [ ] `notes/week9_huatuoGPT_II.md` — 论文精读笔记
- [ ] `notes/week9_bloombergGPT.md` — 论文精读笔记
- [ ] `notes/week9_cpt_concepts.md` — CPT 关键概念整理

---

## 自测题

1. **为什么 CPT 学习率要比预训练小 10-100 倍？**
2. **TAPT 和 DAPT 的区别是什么？各自适用什么场景？**
3. **画出 CPT 数据混合比例 vs 遗忘率的预期 trade-off 曲线**

> 答案: 1) 预训练是在随机初始化的参数上学习，梯度可以很大；CPT 是在已经学好的参数上微调，大学习率会破坏已学到的表示。2) TAPT = 用目标任务的数据做预训练（数据量小但精准）；DAPT = 用目标领域的全部数据做预训练（数据量大但成本高）。TAPT 适合数据少的场景，DAPT 适合数据充足的场景。3) 纯领域 → 遗忘率高、领域提升大；70-30 → 平衡；50-50 → 遗忘率低但领域提升也小。曲线呈反比关系。

---

## 验收清单

- [ ] 3 篇论文精读笔记完成
- [ ] 能回答"为什么 CPT 学习率要比预训练小 10-100 倍？"
- [ ] 能画出 CPT 数据混合比例 vs 遗忘率的预期 trade-off 曲线
- [ ] CPT 关键概念整理完成
