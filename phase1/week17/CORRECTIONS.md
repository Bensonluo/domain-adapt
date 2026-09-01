# Week 17 纠错记录

## C17-01：“真实迁移”证据不足

- 原问题：CMExam 500 题从 51.2% 到 53.4%（11 题）被写成无保留的真实迁移。
- 为什么错：只有一个训练 seed、没有区间，且该 holdout 后续反复用于比较和决策。2026-08-28 规范化审计进一步发现：500 题中 26 题与官方 train 重复，**8 题实际进入 GRPO 8K train**。
- 修复：保留 +2.2pp 历史点估计，但迁移声明改为 `INVALIDATED`；必须在无 train overlap 的冻结确认集、多 seed 和配对区间上重跑。

## C17-02：“reward hacking 几乎不可能/未出现”过强

- 原问题：客观字母正确率和 unparseable=0 只能排除部分格式攻击。
- 修复：改为“未观察到解析失败型或显式标签格式 hacking”；解释质量、校准、标签先验和策略性输出仍需 probes。

## C17-03：GRPO 深度消融未完成

- 原问题：一个正式配置跑通被视为主攻方向完成。
- 修复：机制闭环记 `DONE`；group size、temperature、beta 与 reward ablation 进入 rerun plan。

## C17-04：“没有 reward hacking”缺少可枚举的反证范围

- 为什么错：若不在运行前列出 probes，结果后只展示通过项会形成选择性报告。
- 修复：`phase1/confirmation/grpo.json` 预注册标签先验、纯格式、空解释、错解释配正确标签和不可解析输出五类 probes，并固定训练—确认重叠必须为 0、3 个 seed。只允许表述“排除了实际测试的类型”。

## C17-05：500 题 aggregate 只保留了前 50 条逐题预测

- 原问题：`eval_cmexam.py` 用全部 500 题计算 accuracy，却只写入 `preds[:50]`。因此审计发现 8 道训练重叠后，无法精确重算完整 clean accuracy、paired CI 和 McNemar。
- 为什么错：aggregate 足以展示点估计，但无法支持配对推断、剔除污染题或复核失败类型；“跑过 500 题”不等于“保留了 500 题证据”。
- 可恢复分析：保留前缀中有 1 道 overlap，剔除后 n=49，base=53.06%、GRPO=61.22%、delta=+8.16pp，95% paired bootstrap CI [−2.04,+18.37]pp，McNemar p=0.21875；它只是非随机前 50 题的探索性样本，不代表完整集。
- 完整集边界：总净增 11 题、overlap=8，故剩余 492 题的净增严格位于 3–19 题，即 +0.61pp 至 +3.86pp（前提是历史 aggregate 可信）。方向不能完全由 8 题污染解释，但精确效应和显著性不可恢复，原“真实迁移”仍为 `INVALIDATED`。
- 修复：未来评测全量保存 `index/gold/pred/correct`；确认实验使用无 overlap 候选、多 seed 和完整逐题产物。结构化记录见 `phase1/audit/week17_clean_reanalysis.json`。

## C17-06：probe 只有名字仍允许事后改变判定

- 原问题：首版 GRPO 确认合同列了 5 个 probe 名称，但没有样本来源、确定性变换、评分量和阈值，无法支持“已排除”的措辞。
- 修复：每个 probe 现在固定 source、transform、metric 和 numeric pass threshold；只有五项全部通过才允许有限范围的 exclusion wording。validator 拒绝缺字段或缺阈值的 probe。
- 二次加固：自然语言 source 仍不够唯一，现由 `build_grpo_probe_fixtures.py` 从冻结确认候选按 label 分层、按 immutable id 排序取固定前缀，实际生成 801 条 versioned fixture；generator、fixture、audit 三者 hash 均写入合同。阈值强制为 `operator=lte` 的数值结构，不能用 `null` 或任意字符串占位。
