# Week 16 纠错记录

## C16-01：IPO 评价存在目标同构

- 原问题：IPO 优化 length-normalized / mean-logp 后，又主要用 mean-logp win rate 判断 IPO 成功。
- 为什么错：评价指标与优化目标同构会机械性偏向 IPO，不能证明偏好质量或泛化更好。
- 修复：IPO 仅记为方向性信号；确认实验使用与目标不同构的盲评/任务 outcome，并扩大 grouped holdout。

补充：2026-08-28 审计发现该周沿用的历史 preference split 有 50 个 prompt 组交叉。Week 16 数字不能用于确认泛化，重跑必须使用 grouped v1。

## C16-02：噪声结论外推过度

- 原问题：在模型几乎不泛化、单 seed 的设置下概括“DPO 没有噪声免疫”。
- 修复：只报告该数据和噪声注入方式下的观测；噪声鲁棒性需多强度、多 seed 和独立质量指标。

## C16-03：IPO 的独立 outcome 约束已机器化

- 修复：确认规范明确要求 `independent_outcome_metric=cmexam_accuracy`、grouped prompt overlap=0，并由 `phase1/confirmation/validate_specs.py` 拒绝旧 random split 或目标同构替代指标。历史 Week 16 结果不因此升级。
