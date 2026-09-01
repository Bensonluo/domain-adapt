# Week 18 纠错记录

## C18-01：蒸馏信息量与优越性表述过度简化

- 原问题：hard label 被概括为“1 bit”，合成数据被描述为近乎无限且更便宜，on-policy 被称为通常最好并可超过 teacher。
- 为什么错：标签信息量取决于词表、序列和条件分布；生成、筛选和验证都有成本；on-policy 收益依赖探索、reward、迭代和预算。
- 修复：全部改为条件性假设；本项目 Week 20 实际只做一轮 rejection-sampling SFT，不能代表一般 on-policy distillation 或 RL。
