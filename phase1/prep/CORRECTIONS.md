# Week 8.5 数据准备纠错记录

## C8.5-01：计划与实际被混读

- 原计划：CPT 每比例 1B+ token、SFT 2000–5000 条 A/B/C、teacher/student 生成 2000+ 偏好对。
- 实际：CPT 后续接通约 1414 万 token；Phase 1 没有建立计划中的 SFT 数据闭环；Week 14 改用公开偏好数据 1399 对。
- 修复：本页保留为前置计划，实际状态以 Phase 1 主 README 和 Week 10/14 为准。

## C8.5-02：随机 90/10 切分不足

- 原问题：只写 90/10，没有要求按 source、document 或 normalized prompt 分组。
- 为什么错：同源片段或同 prompt 可跨 train/test，造成泄漏和过高估计。
- 修复：CPT/SFT 按来源文档分组，偏好数据按规范化 prompt/source 分组；切分前后保存 overlap 审计。

## C8.5-03：长度分布与偏差控制回答不同问题

- 修复：长度分布只能说明偏差存在；长度匹配/分层数据和独立 outcome metric 可用于进一步研究偏差的影响。
