# Phase 0 最小重跑计划

## P0：关闭核心方法论问题

### E1：公平的 base vs adapter 配对重跑

- 同一模型权重格式、量化、runtime、prompt/template、temperature=0、max tokens。
- 冻结配置后，在新的 blind test 上运行一次。
- 保存逐题预测；报告 Top-1、格式成功率、paired bootstrap CI、McNemar。
- 至少 3 个训练 seed。

### E2：masking 目标消融

- A：全序列 loss。
- B：assistant content-only，但不监督结束边界。
- C：assistant content + turn-ending token；这是推荐目标。
- 同模型、数据、初始化、步数、batch、LoRA 和解码。
- 指标：领域正确率、格式遵循、prompt echo、截断/不停止率、通用能力变化。

### E3：质量与数量解耦

- 实验前定义质量维度：事实/标签正确性、指令—回答一致性、输出 schema、来源可追溯性、去重/泄漏、难度与覆盖；不得用长度或模型分数单独代替质量。
- 两名审阅者在不知道实验结果的情况下标注候选池并记录分歧；阈值、排除规则和仲裁流程预先冻结。
- 质量效应：从同一候选样本构造配对版本，逐项注入或修复一种可审计缺陷；固定 500 条，匹配主题、难度、长度和 assistant token。结论只外推到被操纵的质量维度。
- 数量效应：从同一已通过质量门槛的分层池构造嵌套的 500/2000/5000 子集，保持来源、主题和难度比例。
- 分别报告固定 epoch 和固定优化 token/step 两种制度。
- 至少 3 个 seed 和置信区间；实验后不得按结果重定义“高质量”。

### E4：真实更新矩阵的 SVD

- 原“Qwen rank 8 捕获 85%–95%”和“rank 8 普遍足够”永久标为 `RETRACTED`，小模型重跑只形成新的、模型特定的 claim，不能复活旧声明。
- 使用可负担 full FT 的小模型，保存 `W_before` 与 `W_after`；若要陈述 Qwen/Gemma 特定结论，必须在对应模型上取得不受低秩约束的更新证据，否则保持撤回。
- 计算 `ΔW_full = W_after - W_before`，不能用 LoRA 自身的 `BA` 证明低秩假设。
- 对 q/k/v/o 与 FFN 多层报告 `E(4/8/16/32)=Σσ_i²/Σσ²`。
- 配套运行相同条件的 base、full FT 与 LoRA rank 4/8/16/32 下游消融；报告各 rank 相对 full FT 的性能差、方差、参数量和计算预算，而不是只看谱能量。

## P1：关闭评估外部效度问题

- 将旧 800 条合成测试降级为 dev。
- 新建 generator-shift 测试：新模板、新噪声、新候选构造、新 seed。
- 增加真实业务盲测，覆盖别名、错别字、多义项、NONE、正确候选缺失。
- 区分 Recall@K、候选内 Top-1、端到端准确率、NONE 检出率和误合并率。

## P2：证据治理

- Week 3 补显存/时间/loss 日志，或标记 `WAIVED`。
- Week 7 将推导文档完成与脱稿复述能力验收分开。
- 外部项目证据固定 commit，补 overlap 审计和运行 manifest。
