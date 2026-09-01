# Phase 1 评估隔离与盲测协议

## 已确认的污染边界

- CMExam 官方 test 及其 500 题子集已参与 Week 17/19/20/21 的比较、选型和叙事，永久降级为 dev。500 题中另有 26 题与官方 train 规范化重复，其中 8 题实际进入 GRPO 8K train，Week 17 迁移结论因此失效。
- CMMLU 已从 Week 12 起反复参与 baseline、比例、beta、方法和蒸馏臂比较，永久作为 regression/dev benchmark。
- 历史偏好 random split 有 50 个 normalized prompt 组跨 train/holdout，历史胜率只能作描述性证据。
- 改文件名、换目录或重新抽取上述来源，均不能恢复盲测资格。

机器登记见 [`benchmark_registry.json`](benchmark_registry.json)，数据级统计见 [`benchmark_split_audit.json`](benchmark_split_audit.json) 和 [`preference_split_audit.json`](preference_split_audit.json)。

## 当前可用但不等于 blind 的确认集

从项目从未用于选模的 CMExam `valid` 中，按结构规则排除了：

- 与 train 重复的规范化题目；
- 与已污染 test 重复的规范化题目；
- 非单选和无效记录；
- valid 内重复题。

剩余 6,305 题冻结为 `confirmation_candidate_v1`。选择过程未查看模型输出，但由于 CMExam 是公开数据且答案本地可见，其状态只能是 `FROZEN_CONFIRMATION_CANDIDATE_NOT_BLIND`。

## 确认运行前必须冻结

1. claim ID 和允许结论措辞。
2. 候选模型/checkpoint hash。
3. 训练数据 hash、seed、超参数和停止规则。
4. primary metric、最小实际重要差异或非劣效 margin。
5. 配对统计方法、失败类型和多重比较处理。
6. 哪些结果会判定支持、拒绝或证据不足。

冻结内容写入单独 run manifest 后，才能运行确认集。结果出来后禁止基于该结果调参再跑同一 claim；若修改方案，必须建立新 claim，并把该集合降级为 dev。

## Phase 1 正式退出仍缺什么

真正 blind test 需要来自未参与本项目决策的新来源/新模板/真实业务样本，并由外部评估器或 label custodian 保管标签。至少需要：

- 医学任务 blind set；
- 通用能力 blind set；
- 数据来源、许可、去重/训练重叠审计；
- 只返回逐题正确性或签名结果、而非在选模阶段暴露标签。

在这些条件满足前，6,305 题确认集可以提高证据强度，但不能把 Phase 1 状态改为 `PASSED`。
