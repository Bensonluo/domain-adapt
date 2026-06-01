# Week 19：Response Distillation 实战

> 目标: 用大模型生成 5000 条医疗 Q&A，训练小模型学习，对比蒸馏 vs 真实数据。
> 预计时间: 12-16 小时

> **思考锚点**: "蒸馏数据训练的小模型，和真实数据训练的，差距在哪？蒸馏的瓶颈是什么？"

---

## Day 1-2: 蒸馏数据生成

### 做什么
1. 用 DeepSeek-V3 / GPT-4o 生成 5000 条医疗 Q&A
2. 同一批问题，用小模型（Qwen2.5-3B）也回答
3. 对比 teacher vs student 回答质量

### 跑
```bash
python phase1/week19/distill_response.py \
    --questions phase1/data/raw/questions.jsonl \
    --teacher deepseek-v3 \
    --student qwen2.5-3b-instruct \
    --n 5000 \
    --output phase1/data/processed/distill_response/
```

---

## Day 3-4: 训练 + 对比

### 做什么
训练 3 个模型对比：
1. **基线**: CPT + SFT（真实数据）
2. **蒸馏**: CPT + SFT（teacher 回答作为训练数据）
3. **混合**: CPT + SFT（真实 50% + 蒸馏 50%）

### 交付物

- [ ] 5000 条蒸馏 Q&A 数据
- [ ] 3 种数据源的对比结果
- [ ] 蒸馏效率分析（API 成本 vs 效果提升）

---

## 验收清单

- [ ] 蒸馏数据生成完成（5000+ 条）
- [ ] 3 种训练方案对比完成
- [ ] 有蒸馏 vs 真实数据的量化差距
