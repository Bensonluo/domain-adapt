# SFT 最佳实践 Checklist

> 来源: Week 5 Day 7 整理；2026-08-28 方法论审查后改为“实验前检查项”，不再把未经本项目验证的经验写成已完成结论。

---

## 数据

- [ ] 将“质量效应”和“数量效应”分开验证；500 条高质量与 5000 条低质量数据的优劣仍需受控实验比较
- [x] 去重 (MinHash / exact match)
- [x] 过滤低质量 (长度 < 10 字、格式错误)
- [x] 格式统一为 OpenAI messages 格式
- [ ] 先按 prompt、实体、文档来源或生成模板分组，再做 train/dev/test；随机 90/10 只适用于不存在组级依赖的数据

## 模型选择

- [x] 基座模型是否有 Instruct 版本? (优先用 Instruct)
- [x] 模型大小 vs 任务复杂度匹配

## Chat Template

- [x] 使用模型对应的 chat_template (Qwen→ChatML, Llama-3→自己的格式)
- [ ] 模板错配有明确风险，但“显著下降”需要同条件训练对照支持
- [x] 用 `tokenizer.apply_chat_template()` 而不是手写拼接

## Loss Masking

- [x] 只在 assistant response 上计算 loss
- [x] multi-turn 对话: 每个 assistant turn 都要保留,其他设为 -100
- [x] assistant 内容及其 turn-ending token（EOS/im_end）参与 loss；prompt、角色前缀和 padding 设为 -100

## 训练超参

| 参数 | QLoRA | 全量微调 |
|------|-------|----------|
| Learning rate | 1e-4–2e-4 起始搜索 | 1e-5–5e-5 起始搜索 |
| Epochs | 1-3 起始范围，以 dev/过拟合信号决定 | 1-3 起始范围，以 dev/过拟合信号决定 |
| Effective batch | 在显存预算内固定并记录 | 在显存预算内固定并记录 |
| Warmup | 1%-5% 起始范围 | 1%-5% 起始范围 |
| LR scheduler | 作为配置记录并受控比较 | 作为配置记录并受控比较 |

## LoRA 配置

- [ ] rank: 4/8/16/32 为常见搜索点；不能仅按模型大小决定
- [ ] alpha: rank 或 2×rank 可作起点，需与 rank/LR 联合验证
- [ ] target_modules: q/v 或 attention projections 可作起点；扩展模块增加容量和成本，不是越多越好
- [ ] dropout: 0-0.1 为常见范围，按数据规模和 dev 结果决定

## 过拟合信号

- [x] train loss 持续下降, val loss 上升 → 过拟合
- [x] 生成文本开始"背诵"训练数据 → 过拟合
- [x] 对策: 减少 epochs, 增加 dropout, 增加数据多样性

## 评估

- [x] 训练前后跑同样的 benchmark,记录 delta
- [ ] 人工测试使用冻结题集、盲化模型身份、明确 rubric，并保存逐题评分
- [ ] 原始基座 vs 微调模型必须使用同 runtime、模板、解码参数和相同题目
- [ ] 至少两名评分者；序数评分按维度报告 weighted kappa 或其他预先指定的一致性指标
