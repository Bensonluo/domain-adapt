# Phase 1 后续实验建议

以下方案分别对应收益归因、训练波动和泛化等问题，可按价值与成本选择；不需要重做所有周。

## 已有数据准备与评估选项

### E0：评估数据使用

- [x] 将既有 CMExam 500 题、完整 test 来源和 CMMLU 标为 dev，并登记 hash。
- [x] 从未用于选模的 CMExam valid 中排除 train/test overlap，冻结 6,305 题确认候选；因标签本地可见，明确不称 blind。
- 若重点是外部泛化，可增加新来源医学与通用样本；独立评估方保管标签是一种可选安排。
- 在 dev 上选择候选、记录配置后，再评估未用于选模的样本，有助于减少选择偏差。
- 保存逐题预测、paired bootstrap CI、McNemar 和失败类型。

### E1：数据泄漏与 lineage 审计

- [x] 偏好数据按规范化 prompt/source 分组重切；新 split 1,299/100，prompt overlap=0。历史 random split 检出 50 个 prompt 组交叉。
- [x] 合成替代确认数据已移除所有阈值内重叠：两臂各 2,000 条、同标签分布、completion token 预算差 0.86%，审计见 `week21_clean_replacement_audit.json`。
- 为每个核心 run 固定 data hash、model revision、seed、config、checkpoint、predictions 和 summary。

## 可进一步回答的问题

### E2：CPT 受控确认

- 固定同一 base、tokenizer、语料、优化 token、lr、步数和评测。
- 比较至少：no-CPT、LoRA-CPT、项目所称 full 模式；先验证实际 trainable parameter 范围。
- 配比实验只改变混合比例，以 3 个 seed 观察波动；当前 50/50 是后续采用的基线。

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
- 使用 `week21_control_real_clean_v1.jsonl` 与 `week21_treatment_synthetic50_clean_v1.jsonl`；历史 `replacement_50.jsonl` 用于复现旧实验。
- control/treatment 各至少 3 seed；非劣效 margin 在训练和评估前固定。
- 若进一步研究临床正确性，可安排临床专业评审并记录分歧；当前非临床 AI 复核主要用于初步检查。

## 如何选择

E1 的数据修订已经完成，可直接为后续实验使用。E2–E6 按待回答的问题选择其一，先检查配置和小规模运行成本，再决定是否扩展；E0 的外部样本评估适合进一步研究跨来源表现。
