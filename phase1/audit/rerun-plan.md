# Phase 1 最小确认实验计划

目标不是重做所有周，而是用最少运行关闭会改变决策的核心问题。

## P0：先修评估协议

### E0：冻结新 blind test

- [x] 将既有 CMExam 500 题、完整 test 来源和 CMMLU 标为 dev，并登记 hash。
- [x] 从未用于选模的 CMExam valid 中排除 train/test overlap，冻结 6,305 题确认候选；因标签本地可见，明确不称 blind。
- [ ] 建立由外部 evaluator/label custodian 管理的新医学 blind test 和通用 blind test。
- 所有候选、超参和 winner 先在 dev 上锁定，再统一运行 blind test 一次。
- 保存逐题预测、paired bootstrap CI、McNemar 和失败类型。

### E1：数据泄漏与 lineage 审计

- [x] 偏好数据按规范化 prompt/source 分组重切；新 split 1,299/100，prompt overlap=0。历史 random split 检出 50 个 prompt 组交叉。
- [x] 合成替代确认数据已移除所有阈值内重叠：两臂各 2,000 条、同标签分布、completion token 预算差 0.86%，审计见 `week21_clean_replacement_audit.json`。
- 为每个核心 run 固定 data hash、model revision、seed、config、checkpoint、predictions 和 summary。

## P1：关闭核心方法问题

### E2：CPT 受控确认

- 固定同一 base、tokenizer、语料、优化 token、lr、步数和评测。
- 比较至少：no-CPT、LoRA-CPT、项目所称 full 模式；先验证实际 trainable parameter 范围。
- 配比实验只改变混合比例，并做至少 3 seed；50/50 在此之前只称操作性基线。

### E3：DPO/IPO 确认

- 增加匹配 SFT/instruct baseline，不再只从 CPT base 直接做偏好训练来推断方法优劣。
- 使用 grouped split 和长度平衡/分层评估。
- DPO vs IPO 使用与训练目标不同构的 outcome metric；全量数据、至少 3 seed。

### E4：GRPO 确认

- [x] 历史证据可恢复部分已重分析：完整 clean delta 只能界定为 +0.61pp 至 +3.86pp；前 50 条预测剔除 overlap 后仍仅是 n=49 非随机探索样本。
- 固定 base 和数据，至少 3 seed；对 group size、temperature、beta 选择最小必要消融。
- reward 指标之外增加独立正确率、格式、校准和解释质量检查。
- 构造标签先验、格式投机、空解释/错误解释等 probes，只报告实际排除的 hacking 类型。

### E5：蒸馏受控确认

- hard CE 与 soft KL 使用相同 prompt/completion 来源、相同 token budget、步数和 seed。
- 预先锁定 `alpha ∈ {0, 0.5, 1}`；温度作为次级因素，避免多臂事后选优。
- 将 Week 20 的方案准确命名为“一轮 rejection-sampling SFT”，不外推到一般 on-policy distillation 或 RL。

### E6：合成替代确认

- 使用修复后的 RNG 重生成；记录原始 response 和逐条 provenance。
- 使用 `week21_control_real_clean_v1.jsonl` 与 `week21_treatment_synthetic50_clean_v1.jsonl`；不得回退到历史 `replacement_50.jsonl`。
- control/treatment 各至少 3 seed；非劣效 margin 在训练和评估前固定。
- 临床正确性由至少两名合格人类评审盲评并报告一致性；AI judge 仅作补充。

## 建议执行顺序

`E0 → E1 → E2 → E3/E4/E5 → E6`。先冻结评估和 lineage，否则昂贵重跑仍会产生不可验收结果。
