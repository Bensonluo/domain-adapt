# ADR-P1-001：采用跨周证据注册表与独立盲测

- 状态：Accepted
- 日期：2026-08-28

## 背景

Phase 1 已完成 Week 9–21 的大部分学习和实验动作，但“训练跑完、产物存在、点估计更高、研究声明成立”曾被混作同一种完成。CMExam 500 题和 CMMLU 子集又被跨周反复查看并用于选方案，已不再是独立测试集。

## 备选方案

1. 只修改几处结论文字。成本最低，但不能解决跨周证据漂移。
2. Markdown 治理文件 + 机器可读 artifact lineage。既保留学习历史，又能约束研究声明。
3. 立即把所有状态改造成由 YAML/JSON 自动生成。最严格，但治理工程量超过当前收益。

## 决策

采用方案 2：

- `phase1/README.md` 是唯一阶段状态入口。
- 每条核心研究声明必须有 claim ID，并映射到数据、配置、checkpoint、逐题预测和 summary。
- JSON summary 是证据，不自动等于结论已验证。
- 当前 CMExam 500 题和反复使用的 CMMLU 子集统一降级为 development benchmark。
- 新 blind test 在方案和超参数冻结后只运行一次，结果不得用于继续选模。
- 状态使用 `UNVERIFIED / SUPPORTED_TREND / VERIFIED / RETRACTED / REPLACED / WAIVED`。
- 历史错误保留在各周 `CORRECTIONS.md`，不通过删除历史记录获得通过。
- 客观结构化任务以逐题配对统计为主；只有开放式质量或临床正确性声明才强制人工盲评、双评审和一致性报告。

## 影响

Phase 1 当前状态为 `REMEDIATION_REQUIRED`。既有实验可作为学习产物和探索性证据保留，但 CPT、DPO/IPO、GRPO、蒸馏和合成数据的强因果或普适结论必须经过最小确认实验后才能进入阶段总结。
