# Week 12：CPT 实验（下）+ 灾难性遗忘量化

> 目标: 跑数据混合 ablation，量化灾难性遗忘，找出最优配置。
> 预计时间: 14-20 小时

> **思考锚点**: "CPT 的收益和损失之间，最优的 trade-off 点在哪？"

---

## Day 1-3: 数据混合 Ablation

### 做什么
1. 跑 3 种混合比例（纯领域 / 70-30 / 50-50）
2. 每种配置用相同超参，只改数据
3. 记录每个模型的训练 loss 曲线

### 跑
```bash
# 70-30 混合
python phase1/week11/train_cpt.py --config phase1/week12/cpt_70_30.yaml --output phase1/results/week12_cpt_70_30/

# 50-50 混合
python phase1/week11/train_cpt.py --config phase1/week12/cpt_50_50.yaml --output phase1/results/week12_cpt_50_50/
```

---

## Day 4-5: 评估 + 遗忘量化

### 做什么
1. 每个模型跑医疗 benchmark + 通用 MMLU
2. 计算遗忘率 = (MMLU_before - MMLU_after) / MMLU_before
3. 对比：CPT 后 SFT vs 不做 CPT 直接 SFT
4. 找出最优混合比例

### 跑
```bash
# 医疗 benchmark（测领域提升）
python phase1/utils/eval_benchmark.py --model phase1/results/week12_cpt_70_30/ --tasks medical

# 通用 benchmark（测遗忘）
python phase1/week12/eval_forgetting.py --baseline Qwen/Qwen2.5-3B --finetuned phase1/results/week12_cpt_70_30/
```

---

## 交付物

- [ ] `results/week12_cpt_ablation.md` — CPT 完整实验报告
- [ ] 4 种配置的 loss 曲线对比
- [ ] 遗忘率 vs 领域提升 trade-off 图
- [ ] 最优配置推荐

---

## 自测题

1. **纯领域 CPT 的遗忘率大约是多少？70-30 呢？**
2. **CPT 后做 SFT vs 不做 CPT 直接 SFT，哪个效果好？为什么？**
3. **如果遗忘率 > 20%，你会怎么调整？**

---

## 验收清单

- [ ] 3 种混合比例实验完成
- [ ] 每个模型都有医疗 benchmark + MMLU 分数
- [ ] 遗忘率计算完成
- [ ] 找到最优混合比例 + 写出理由
- [ ] 至少 1 个关于 CPT 的个人 insight
