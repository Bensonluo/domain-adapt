# Phase 1 阶段总结（整改版）

## 当前判定

`REMEDIATION_REQUIRED`

Phase 1 已完成从 CPT、偏好优化、GRPO、蒸馏到合成数据的完整学习链和大量实验产物，尤其 Week 20/21 的结构化 summary、validator 和 lineage 已具备较好基础。但这代表“学习与执行完成度高”，不代表所有研究性结论已通过。

## 目前可以保留的结论

1. **CPT 工程链路已跑通**：小规模真实语料、混合比例和评测管线均有产物。现有规模不支持“完成 1B–3B token 严肃 CPT”（P1-C01）。
2. **新 LoRA-CPT 配置在 development benchmark 上出现正 gain**：三比例点估计为 +0.033~+0.043，所测通用子集未下降。由于同时换模型、方法和步数，不能把改善归因给 LoRA；50/50 只是操作性基线（P1-C02/P1-C03）。
3. **DPO/IPO 已暴露偏好数据长度与泛化问题**：beta sweep 没有支持统计可区分的最优 beta；IPO 有方向性信号，但 outcome 与训练目标同构，需独立确认（P1-C04–P1-C06）。
4. **GRPO 优化链路已跑通，但确认性迁移证据失效**：CMExam development 点估计 +2.2pp；8/500 题实际进入 GRPO 训练，且历史只保留前 50 条逐题预测。由 aggregate 可严格界定剩余 492 题净增为 +0.61pp 至 +3.86pp，但精确 clean 效应、CI 和 McNemar 不可恢复。当前只可保留训练 reward 上升、clean 点估计方向的条件性边界及未观察到解析失败型 hacking（P1-C07）。
5. **蒸馏实验产生了值得确认的趋势**：teacher explanation、soft-KL 和 `kd_pure` 在部分 development 指标上更稳定或更高，但单 seed、多臂选择和控制差异不支持普适机制结论（P1-C08–P1-C11）。
6. **合成数据替代的点估计有希望，但非劣效未成立**：50% 替代相对 control 为 −0.8pp，95% CI [−3.4,+1.8]pp 跨越 −2pp margin；30 条审核仅是 AI 非临床 sanity check（P1-C12/P1-C13）。

## 最重要的方法论收获

- 多个超参变体不等于多个独立 seed。
- 同一 holdout 在被反复查看并用于决策后，就应降级为 dev。
- 点估计 winner 只能成为确认实验候选，不能直接称“最优”。
- 结果随多项变量同时改善时，不能把原因归给其中一个变量。
- 与训练目标同构的评价指标适合诊断优化过程，不足以独立证明真实质量。
- “没有观察到某类失败”只允许排除已测试的失败类型。

## 2026-08-28 数据隔离审计

- 历史 preference split 行级 overlap=0，但 normalized prompt-group overlap=50；因此 Week 15/16 偏好泛化证据被污染。新 grouped v1 已实现 1299/100，prompt overlap=0。
- CMExam 官方 split 不是题目级互斥：train–valid 248、train–test 253、valid–test 54 个规范化题目重叠。
- 历史 CMExam 500 题中 26 题与官方 train 重复，其中 8 题实际进入 GRPO 8K train；Week 17 的迁移 claim 改为 `INVALIDATED`。
- Week 17 的前 50 条逐题预测中只有 index 33 为训练重叠；剔除后 n=49 的探索性 delta 为 +8.16pp，CI 跨 0。由于其余 450 条预测未落盘，该样本不可代替完整 clean 重算。
- 从未用于选模的 valid 中排除所有已知 Phase 1 训练/test 重叠后，冻结 6,305 题确认候选，规范化交叉为 0。但它是公开且标签本地可见的数据，不代替外部 blind test。
- Week 21 已建立 clean matched confirmation v1：两臂各 2,000 条、标签一致、completion token 预算差 0.86%、对确认候选重叠为 0。该修复只关闭未来重跑的数据设计问题，历史非劣效结论不变。

## 下一步

数据 lineage、确认候选和五类运行前合同已经冻结。下一步按 [`../audit/rerun-plan.md`](../audit/rerun-plan.md) 配置外部 evaluator/label custodian，并执行 CPT、DPO/IPO、GRPO、蒸馏和合成替代的多 seed 确认实验。正式退出条件见 [`../audit/exit-gate.md`](../audit/exit-gate.md)。
