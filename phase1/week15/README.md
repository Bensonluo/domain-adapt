# Week 15：DPO 实验（上）

> 目标: 在 CPT+SFT 模型上跑 DPO，测试 3 个 beta 值。
> 预计时间: 12-16 小时

> **思考锚点**: "DPO 的 beta 参数控制什么？beta 太大和太小各有什么后果？"

---

## Day 1-3: DPO 训练（3 个 beta）

### 做什么
1. 加载 CPT+SFT 后的模型
2. 分别用 beta=0.1, 0.3, 0.5 跑 DPO
3. 监控 chosen/rejected reward margin

### 跑
```bash
# beta=0.1
python phase1/week15/train_dpo.py --beta 0.1 --output phase1/results/week15_dpo_0.1/

# beta=0.3
python phase1/week15/train_dpo.py --beta 0.3 --output phase1/results/week15_dpo_0.3/

# beta=0.5
python phase1/week15/train_dpo.py --beta 0.5 --output phase1/results/week15_dpo_0.5/
```

---

## Day 4-5: 初步评估

### 做什么
1. 医疗 benchmark 评估
2. 人工 20 题测试
3. 分析 reward margin 趋势

### 交付物

- [ ] 3 个 DPO 模型训练记录
- [ ] Reward margin 对比图
- [ ] 初步评估结果

---

## 验收清单

- [ ] 3 个 beta 值的 DPO 训练完成
- [ ] Reward margin 曲线正常（chosen > rejected）
- [ ] 初步 benchmark 评估完成
