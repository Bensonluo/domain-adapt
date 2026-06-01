# Week 21：合成数据生成 + 数据质量评估

> 目标: 实现 Self-Instruct + Evol-Instruct pipeline，评估合成数据质量。
> 预计时间: 14-20 小时

> **思考锚点**: "合成数据能替代 50% 真实数据吗？质量损失有多大？"

---

## Day 1-2: Self-Instruct Pipeline

### 做什么
1. 准备 100 条种子指令
2. 大模型生成新问题 + 回答
3. 过滤低质量生成

### 跑
```bash
python phase1/week21/self_instruct.py \
    --seeds phase1/data/raw/seed_instructions.jsonl \
    --model gpt-4o \
    --n 5000 \
    --output phase1/data/processed/synthetic/
```

---

## Day 3: Evol-Instruct

### 做什么
1. 实现问题复杂度演化：简单 → 复杂
2. 多步演化（depth 1-3）

### 跑
```bash
python phase1/week21/evol_instruct.py \
    --input phase1/data/processed/synthetic/self_instruct.jsonl \
    --model gpt-4o \
    --output phase1/data/processed/synthetic/evolved/
```

---

## Day 4-5: 质量评估 + 替代实验

### 做什么
1. 合成数据质量评估：
   - 多样性（BERTScore / Self-BLEU）
   - 正确性（抽样人工审核）
   - 与真实数据的分布差异
2. **关键实验**: 合成数据替代 50% 真实数据，看效果是否保持

### 交付物

- [ ] Self-Instruct pipeline 代码
- [ ] Evol-Instruct pipeline 代码
- [ ] `results/week21_synthetic_quality.md` — 质量评估报告
- [ ] `results/week21_replacement_experiment.md` — 替代实验结果

---

## 自测题

1. **Self-Instruct 生成的指令质量受什么因素影响最大？**
2. **Evol-Instruct 的"演化"在做什么？为什么不直接生成复杂问题？**
3. **合成数据替代 50% 真实数据后，效果掉了多少？可以接受吗？**

---

## 验收清单

- [ ] Self-Instruct pipeline 完成
- [ ] Evol-Instruct pipeline 完成
- [ ] 合成数据质量评估完成
- [ ] 替代实验完成
- [ ] 有合成 vs 真实数据的量化对比
