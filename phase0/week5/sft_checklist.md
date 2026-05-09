# SFT 最佳实践 Checklist

> 来源: Week 5 Day 7 整理的个人 SFT 经验

---

## 数据

- [ ] 质量 > 数量: 500 条好的 > 5000 条差的
- [ ] 去重 (MinHash / exact match)
- [ ] 过滤低质量 (长度 < 10 字、格式错误)
- [ ] 格式统一为 OpenAI messages 格式
- [ ] train/test split: 90/10

## 模型选择

- [ ] 基座模型是否有 Instruct 版本? (优先用 Instruct)
- [ ] 模型大小 vs 任务复杂度匹配

## Chat Template

- [ ] 使用模型对应的 chat_template (Qwen→ChatML, Llama-3→自己的格式)
- [ ] 模板错配会导致性能显著下降
- [ ] 用 `tokenizer.apply_chat_template()` 而不是手写拼接

## Loss Masking

- [ ] 只在 assistant response 上计算 loss
- [ ] multi-turn 对话: 每个 assistant turn 都要保留,其他设为 -100
- [ ] 特殊 token (eos, im_end) 也设为 -100

## 训练超参

| 参数 | QLoRA | 全量微调 |
|------|-------|----------|
| Learning rate | 2e-4 | 5e-5 |
| Epochs | 1-3 | 1-3 |
| Batch size | 尽可能大 | 尽可能大 |
| Warmup | 3% | 3% |
| LR scheduler | cosine | cosine |

## LoRA 配置

- [ ] rank: 8 (小模型) / 16 (大模型), 32 通常没必要
- [ ] alpha: = rank 或 2x rank
- [ ] target_modules: ["q_proj", "v_proj"] 最少,加更多更好
- [ ] dropout: 0.05-0.1

## 过拟合信号

- [ ] train loss 持续下降, val loss 上升 → 过拟合
- [ ] 生成文本开始"背诵"训练数据 → 过拟合
- [ ] 对策: 减少 epochs, 增加 dropout, 增加数据多样性

## 评估

- [ ] 训练前后跑同样的 benchmark,记录 delta
- [ ] 人工测试 20+ 个领域问题
- [ ] 对比: 原始基座 vs 微调后的回答
