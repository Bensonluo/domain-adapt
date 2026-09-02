# Week 14 纠错记录

## C14-01：长度偏差方向解释错误

- 原问题：将“chosen 更长”简单解释为序列 `Σ logp` 更大，因此 longer 自动获胜。
- 为什么错：token log-prob 通常为负，序列更长往往使 sum 更负；长度偏差确实存在，但方向由长度、token 难度、mask 和 loss 共同决定。
- 修复：保留 93.5% 的数据偏差事实，撤回错误符号推导；用长度匹配分层和独立 outcome metric 实测。

## C14-02：随机切分没有阻止 prompt 泄漏

- 原问题：允许同一 prompt 的不同三元组保留，随后按样本随机切分，可能使同 prompt 跨 train/holdout。
- 实测：2026-08-28 审计发现历史 train/holdout 有 **50 个 normalized prompt 组交叉**；1399 行只有 1061 个 prompt 组。
- 修复：已生成 `train_grouped_v1.jsonl`（1299）和 `dev_grouped_v1.jsonl`（100），prompt group overlap=0；审计见 `../audit/preference_split_audit.json`。历史结果保留原数据背景，后续设计采用新版本。

## C14-03：CPT-only 基线限制

- 原问题：直接从 base/CPT-only 做 DPO 后，可能把“缺少 instruction/SFT prior”误解为 DPO 本身失败。
- 修复：探索结果保留；增加匹配的 SFT/instruct baseline 有助于区分继续训练与偏好优化的效果。
