# Week 16：DPO 实验（下）+ 对比 + 失败模式

> 目标: 完成 DPO 完整评估，设计搞坏实验，分析失败模式。
> 预计时间: 14-20 小时

> **思考锚点**: "DPO 在什么条件下会失败？失败的表现是什么？"

---

## Day 1-2: 完整对比评估

### 做什么
1. Benchmark 分数：DPO 3 个 beta vs 基线
2. 人工评估 50 题
3. LLM-as-judge 对比
4. 生成长度分析（DPO 的长度偏差）
5. 训练效率对比

### 跑
```bash
python phase1/week16/compare_dpo.py \
    --models phase1/results/week15_dpo_0.1 phase1/results/week15_dpo_0.3 phase1/results/week15_dpo_0.5 \
    --baseline phase1/results/week11_cpt_pure/ \
    --output phase1/results/week16_dpo_comparison/
```

---

## Day 3-4: 搞坏实验（Failure Mode 分析）

### 做什么
1. 偏好数据加 30% 噪声 → 看 DPO 崩溃程度
2. chosen/rejected 长度差异巨大 → 看长度偏差
3. beta 极小 (0.01) 和极大 (10) → 看极端情况

### 跑
```bash
# 噪声实验
python phase1/week15/train_dpo.py --beta 0.3 --noise 0.3 --output phase1/results/week16_dpo_noisy/

# 极端 beta
python phase1/week15/train_dpo.py --beta 0.01 --output phase1/results/week16_dpo_beta_0.01/
python phase1/week15/train_dpo.py --beta 10.0 --output phase1/results/week16_dpo_beta_10/
```

---

## Day 5: IPO/KTO 快速了解

### 做什么
- 读 IPO/KTO 论文各 30 分钟（理解核心思想，不跑实验）
- 整理到对比表中

---

## 交付物

- [ ] `results/week16_dpo_comparison.md` — DPO 完整对比报告
- [ ] 搞坏实验结果
- [ ] 失败模式分析

---

## 验收清单

- [ ] DPO 3 个 beta 完整评估完成
- [ ] 搞坏实验完成（至少 2 个）
- [ ] 失败模式分析文档完成
- [ ] 至少 1 个关于 DPO 失败模式的个人 insight
