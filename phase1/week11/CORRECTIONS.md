# Week 11 纠错记录

## C11-01：“全量 CPT”命名不准确

- 原问题：`full / num_layers=-1` 的 run 被称为全参数 CPT，但记录显示 trainable 约 66%（498M/752M）。
- 为什么错：框架的 `full` 模式不必然等于 embedding、head 与全部参数都训练。
- 修复：历史 run 改称“MLX full 模式 / 全 transformer layers”；具体训练范围可从 trainable parameter 清单确认。

## C11-02：demo loss 被用于方法结论

- 原问题：15 条 train、1 条 val、200 iter 的结果被用于解释过拟合和方法优劣。
- 为什么错：单条验证样本方差极高，无法支持稳定结论。
- 修复：该 run 完成了训练流程验证；单条验证样本的 loss 变化尚不足以比较稳定泛化或方法优劣。
