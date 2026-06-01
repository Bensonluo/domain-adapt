# Week 17：GRPO 实战（主攻深度方向）

> 目标: 深度实验 GRPO，设计 reward function，与 DPO 完整对比。
> 预计时间: 16-24 小时

> **这是 Phase 1 的两个主攻深度方向之一。按深度执行方法论跑 3+ 变体。**
>
> **思考锚点**: "reward 上升但质量真的变好了吗？怎么区分真实提升和 reward hacking？"

---

## Day 1-2: GRPO 训练 + Reward Function 设计

### 做什么
1. 在 CPT+SFT 后的模型上跑 GRPO
2. 设计 domain-specific reward function

### Reward Function 设计
```python
# 方案 A：rule-based
# - 格式正确性 + 关键词覆盖 + 长度惩罚

# 方案 B：小模型打分
# - 用另一个 LLM 做 answer quality 评分
```

### 跑
```bash
python phase1/week17/train_grpo.py \
    --model phase1/results/week11_cpt_pure/ \
    --reward rule_based \
    --output phase1/results/week17_grpo_rule/
```

---

## Day 3-4: 深度实验（3+ 变体）

### 变体 1：num_generations
- 测试 4 / 8 / 16 个候选
- 观察 group diversity 和最终质量的关系

### 变体 2：temperature
- 测试 0.7 / 0.9 / 1.2
- 观察生成多样性的影响

### 变体 3：KL 系数
- 测试不同 KL 系数对 reward hacking 的控制效果

### 变体 4：reward function ablation
- Rule-based vs 小模型打分 vs 混合

---

## Day 5: Reward Hacking 深度分析

### 做什么
1. 每 100 步抽样 10 条生成，人工评分
2. 如果 reward 上升但人工评分下降 → reward hacking
3. 标记 hacking 出现的 step
4. 分析原因 + 尝试修复

### 深度实验模板
```
Step 1: 跑 GRPO（baseline config），记录 reward 曲线 + 生成质量
Step 2: 问："reward 上升但质量真的变好了吗？"
Step 3: 诊断实验：每 100 步抽样评分
Step 4: 如果发现 hacking → 分析原因
  - 假设 A：reward function 有漏洞
  - 假设 B：KL 约束太弱
Step 5: 用实验验证假设
Step 6: 记录结论 → 这是你的 insight，论文里没有
```

---

## 交付物

- [ ] `results/week17_grpo_experiments.md` — GRPO 训练记录
- [ ] `reward_functions.py` — reward function 设计
- [ ] `results/week17_grpo_vs_dpo.md` — 与 DPO 对比分析
- [ ] 超参数直觉数据库更新

---

## 自测题

1. **GRPO 的 num_generations 从 4 增加到 16，效果一定变好吗？为什么？**
2. **你怎么判断出现了 reward hacking？具体用什么指标？**
3. **GRPO vs DPO 在相同数据上，哪个效果更好？为什么？**

---

## 验收清单

- [ ] GRPO 训练完成（至少 3 个变体）
- [ ] Domain-specific reward function 设计完成
- [ ] GRPO vs DPO 完整对比数据
- [ ] 至少 1 个关于 reward hacking 的个人发现
- [ ] Reward function trade-off 分析
- [ ] 能不查资料说出 GRPO 最佳的 3 个超参值
