# Week 11：CPT 实验（上）

> 目标: 在 Qwen2.5-3B base model 上跑第一次 CPT，监控训练过程。
> 预计时间: 10-14 小时

> **思考锚点**: "CPT 的 loss 曲线和 SFT 的 loss 曲线有什么不同？为什么？"

---

## Day 1-2: 配置 + 启动训练

### 做什么
1. 加载 Qwen2.5-3B（纯 base model，非 Instruct）
2. 配置 CPT 训练参数
3. 跑第一个实验：纯医疗语料 CPT

### 跑
```bash
python phase1/week11/train_cpt.py \
    --model Qwen/Qwen2.5-3B \
    --data phase1/data/processed/cpt/ \
    --config phase1/week11/cpt_config.yaml \
    --output phase1/results/week11_cpt_pure/
```

### 训练配置参考
```yaml
# cpt_config.yaml
learning_rate: 1e-5           # 比预训练小 10-100 倍
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
max_steps: 20000
warmup_steps: 500
lr_scheduler_type: cosine
logging_steps: 50
save_steps: 1000
bf16: true
```

---

## Day 3-5: 监控 + 分析

### 做什么
1. 监控训练 loss 曲线
2. 定期计算验证 perplexity
3. 初步生成测试（观察领域语言能力变化）

### 交付物

- [ ] `results/week11_cpt_pure/` — 训练记录 + loss 曲线
- [ ] 验证 perplexity 变化记录
- [ ] 初步生成测试结果

---

## 验收清单

- [ ] CPT 训练完成，loss 稳定下降
- [ ] 验证 perplexity 低于基线
- [ ] Loss 曲线截图保存
