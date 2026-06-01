# Week 14：DPO + GRPO 实战准备

> 目标: 构建偏好数据集，阅读 TRL 源码，设计实验矩阵。
> 预计时间: 12-16 小时

> **思考锚点**: "偏好数据的质量怎么保证？chosen 真的比 rejected 好吗？"

---

## Day 1-2: 偏好数据集构建

### 做什么
1. 用 prep 阶段生成的偏好数据
2. 人工抽检 100 对，确认 chosen 确实优于 rejected
3. 分析长度偏差

### 跑
```bash
python phase1/prep/build_preference_data.py \
    --questions phase1/data/raw/questions.jsonl \
    --n 2000 \
    --output phase1/data/processed/preference/
```

---

## Day 3-4: TRL 源码阅读

### 做什么
1. 阅读 `trl/trainer/dpo_trainer.py` — 理解 DPO loss 实现
   - 关注: `dp_loss` 函数、chosen/rejected log prob 计算
2. 阅读 `trl/trainer/grpo_trainer.py` — 理解 GRPO 实现
   - 关注: reward normalization、generation 策略

### 关键代码路径
- https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

---

## Day 5: 实验矩阵设计

### 做什么
设计 6 组实验：

| # | 配置 | 目标 |
|---|------|------|
| 基线 | CPT + SFT（不加对齐） | 对照组 |
| 实验 1 | CPT + SFT + DPO (beta=0.1) | 弱对齐 |
| 实验 2 | CPT + SFT + DPO (beta=0.3) | 中等对齐 |
| 实验 3 | CPT + SFT + DPO (beta=0.5) | 强对齐 |
| 实验 4 | CPT + SFT + GRPO (baseline reward) | GRPO 基线 |
| 实验 5 | CPT + SFT + GRPO (不同 reward) | Reward function ablation |

---

## 交付物

- [ ] 偏好数据集（2000+ 对）+ 质量报告
- [ ] TRL 源码阅读笔记
- [ ] 实验设计文档

---

## 验收清单

- [ ] 偏好数据构建完成，长度偏差可控
- [ ] DPOTrainer + GRPOTrainer 源码阅读完成
- [ ] GRPO reward function 设计完成
- [ ] 实验矩阵文档完成
