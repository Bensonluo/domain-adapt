# Week 19 纠错记录

## C19-01：机制解释超过实验控制

- 原问题：由三个单 seed 臂直接断言“人写解释砸知识、teacher 解释不砸”“mixed 稀释”。
- 为什么错：解释来源同时改变长度、风格、信息密度、正确性和难度；小幅差异也在噪声内。
- 修复：这些结论降为 `SUPPORTED_TREND` 或机制假设；需同题、同长度/token budget、多 seed 的配对控制。

## C19-02：共享 holdout 已成为 dev

- 原问题：CMExam 500 题在 Week 17 后继续用于 Week 19 选择和叙事。
- 修复：该集合统一称 development benchmark；新 blind test 冻结后只运行一次。

## C19-03：hard/soft 比较没有隔离 completion 来源

- 为什么错：历史 hard、teacher response 与 soft logits 跨周使用了不同 completion/筛选链路，不能把差异归因于监督形式。
- 修复：确认规范要求 hard CE、mixed 与 pure KL 使用相同 prompt/completion ID、相同训练 token/步数和 3 个 seed；Week 19 机制解释继续只作为假设。
