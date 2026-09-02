# Week 15 纠错记录

## C15-01：beta=0.1“最优”无统计依据

- 原问题：脚本依据 matched bucket 3/13 vs 2/13 选择 beta=0.1。
- 为什么错：n=13 的一题差异不足以区分三个 beta，测试集参与选优；后续审计又发现历史 train/holdout 有 50 个 normalized prompt 组交叉。
- 修复：beta=0.1 只保留为点估计候选；原“最优 beta”声明撤回。

## C15-03：历史 split 泄漏已证实

- 实测：旧 `train_split.jsonl` / `holdout.jsonl` 行级 overlap=0，但 prompt-group overlap=50。
- 修复：旧文件仅用于历史复现；未来 DPO/IPO 使用 `train_grouped_v1.jsonl` / `dev_grouped_v1.jsonl`，对应 hash 和审计固定在 `phase1/audit/preference_split_audit.json`。

## C15-02：“没有灾难性遗忘”证据范围过宽

- 原问题：单 seed、少量 CMMLU 子集的微小变化被写成主安全结论。
- 修复：当前 dev 子集未观察到明显下降；更多 seed 和新样本可进一步研究稳定性，本次评估没有覆盖整体安全性。

## C15-04：DPO 确认设计缺少匹配 SFT baseline

- 为什么错：直接从 CPT base 比偏好模型，无法区分“继续训练”与“偏好目标”的作用。
- 修复：`phase1/confirmation/dpo_ipo.json` 固定 grouped v1、匹配 SFT control、DPO/IPO 两个 treatment 和 3 个 seed；主指标改为与 preference logp 不同构的 CMExam 准确率。

## C15-05：只锁 base config 不能唯一识别起始模型

- 原问题：首版合同记录了上游 revision 和 `config.json` hash，但不同适配权重可以共享同一 config。
- 修复：DPO/IPO、GRPO、KD 和 synthetic 四个合同同时锁定 Week 12 fused `model.safetensors` SHA-256；validator 要求下游合同必须有可校验 weights hash。历史泄漏限制也由模糊的 `unverified` 改为实测 `historical_prompt_group_overlap_50`。
