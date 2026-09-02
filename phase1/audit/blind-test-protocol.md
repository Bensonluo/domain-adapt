# Phase 1 评估数据与后续验证

## 当前评估数据的使用情况

- CMExam 官方 test 及其 500 题子集参与了 Week 17/19/20/21 的比较与选型，按实际用途记录为 dev。500 题中有 26 题与官方 train 规范化重复，其中 8 题实际进入 GRPO 8K train，影响了迁移效果的解释。
- CMMLU 从 Week 12 起参与 baseline、配比、beta 和方法比较，用于 development regression。
- 历史偏好 random split 有 50 个 normalized prompt 组跨 train/holdout，已有结果反映该切分下的表现。
- 同一批数据此前参与过哪些选择，会影响结果的独立性；更换文件名或目录不会改变使用历史。

机器索引见 [`benchmark_registry.json`](benchmark_registry.json)，统计见 [`benchmark_split_audit.json`](benchmark_split_audit.json) 和 [`preference_split_audit.json`](preference_split_audit.json)。

## 已准备的本地确认集

从未用于本项目选模的 CMExam `valid` 中，排除了与 train/test 重复的题目、非单选或无效记录及内部重复题，得到 6,305 题 `confirmation_candidate_v1`。

筛选未使用模型输出，但公开数据的答案本地可见，因此记录为 `FROZEN_CONFIRMATION_CANDIDATE_NOT_BLIND`。它可以用于研究已选方案在另一批样本上的表现。

## 后续实验如何减少歧义

提前记录待回答的问题、候选模型、数据版本、seed、超参数、主指标和统计方法，有助于区分事前设计与结果出来后的解释。已有 run manifest 模板可直接用于这一记录。

如果结果用于继续调参，就将这次使用记为探索过程。重复实验与新的评估样本可帮助判断发现是否稳定。

## 外部验证的适用场景

若后续关注跨来源泛化或真实业务效果，可以增加新来源、新模板或业务样本。由独立评估方保管标签是减少选模影响的一种方式，并非所有实验都需要这一安排。临床正确性则需要相应专业评审；当前非临床 AI 抽检没有回答这一问题。
