# Phase 0 评估报告模板

## 1. Benchmark 评估

### 基座模型 (Qwen2.5-3B-Instruct)

| Task | Accuracy | Acc (norm) |
|------|----------|------------|
| mmlu_clinical_knowledge | | |
| mmlu_anatomy | | |
| mmlu_medical_genetics | | |
| **Average** | | |

### SFT 后模型 (domain-sft-merged)

| Task | Accuracy | Acc (norm) | Δ |
|------|----------|------------|---|
| mmlu_clinical_knowledge | | | |
| mmlu_anatomy | | | |
| mmlu_medical_genetics | | | |
| **Average** | | | |

### 分析

TODO:
- [ ] 哪些 task 提升了? 提升了多少?
- [ ] 哪些 task 下降了? 为什么? (灾难性遗忘?)
- [ ] 平均提升/下降多少?

---

## 2. LLM-as-Judge

| 比较对象 | A 获胜 | B 获胜 | 平局 |
|----------|--------|--------|------|
| Base vs SFT | | | |

TODO:
- [ ] 分析 judge 的 bias (位置 bias、长度 bias)
- [ ] 哪些类型的问题 SFT 后明显改善?

---

## 3. 人工评估

### 评分统计

| 模型 | 准确性 | 完整性 | 安全性 | 可读性 | 总分 |
|------|--------|--------|--------|--------|------|
| Base | | | | | |
| SFT | | | | | |

### IAA (Cohen's Kappa)

TODO:
- [ ] 评分者 1 vs 评分者 2 的 Kappa
- [ ] 是否 > 0.6?

---

## 4. 综合结论

TODO: 总结 Phase 0 训练的效果
- [ ] 领域适配是否成功?
- [ ] 有哪些意外的发现?
- [ ] 下一步 (Phase 1) 的优化方向?
