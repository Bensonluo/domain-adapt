# Week 21 纠错记录

## C21-01：点估计通过不等于统计非劣效成立

- 原问题：treatment 比 control 低 0.8pp，满足点估计阈值，但 CI [−3.4,+1.8]pp 跨越 −2pp margin。
- 修复：正式结论保持“有希望但证据不足”；`P1-C12` 标 `INVALIDATED`，不能称 50% 替代已非劣。

## C21-02：评测集和 control 复用

- 原问题：复用 Week 19 checkpoint 与跨周反复查看的 CMExam 500 题，无法承担最终盲测。
- 修复：现有结果为开发性证据；确认实验重训 control/treatment 多 seed，并使用新 blind test。

## C21-03：泄漏、复现与审核范围

- 原问题：替换集保留 12 条高相似真实来源题；历史 raw 生成早于 RNG 修复；30 条为 Codex AI 非临床复核。
- 修复：确认实验移除或预先分层 overlap，使用新 RNG 重生成；AI 复核只称 sanity check，临床正确性需合格人类盲评。

## C21-04：交付物名称漂移

- 原问题：README 勾选两个并不存在的 Markdown 报告，实际核心产物为 `week21_summary.json` 等结构化证据。
- 修复：验收以 artifact manifest 中存在且可校验的路径为准；不存在的计划文件不再视为已交付。

## C21-05：原“等规模替代”没有保证等标签、近 token 预算和评估隔离

- 原问题：历史 treatment 虽为 1,000 real + 1,000 synthetic，但替换不是嵌套配对；保留了 12 条与评测高相似的真实题，标签分布和有效训练 token 也未作为不变量锁定。错误根因是把“样本条数相同”误当成“唯一处理变量只有数据来源”。
- 修复：新增 `prepare_clean_matched_replacement.py`。control/treatment 使用相同 slot；合成题按答案标签和 completion token 长度匹配被替换真实题；所有训练题对冻结确认候选执行 3-gram Jaccard `<0.78`；审计固定输入/输出 hash。
- 验证：两臂各 2,000 条，标签计数完全一致，completion token 预算相差 0.86%，两臂 prohibited overlap 均为 0；`phase1/week21/tests` 覆盖确定性、预算和泄漏拒绝。
- 边界：这关闭了数据设计缺陷，没有重写历史结果，也不等于确认实验已完成。多 seed 同配置重训和外部 blind test 仍未执行。

## C21-06：非劣效 margin 和判定过去没有形成运行前合同

- 为什么错：只看 −0.8pp 点估计后再解释阈值，会混淆操作性筛选与正式非劣效推断。
- 修复：`phase1/confirmation/synthetic_replacement.json` 在重训前固定 margin=`−0.02`、3 个 seed、单一主比较及“95% CI 下界必须高于 margin”的判定；同时锁定 clean v1 两臂 hash。外部 blind 和两名合格盲评者仍是退出条件。

## C21-07：多 seed 与逐题配对的合并规则不明确

- 原问题：首版只写“aggregate paired CI”，没有说明如何同时处理训练 seed 变异和同题配对；“无无法解释的反转”还是主观条件。
- 修复：预注册两层 hierarchical paired bootstrap：外层有放回抽 3 个训练 seed，内层在每个 seed 内有放回抽 6,305 个配对题目，重复 10,000 次并对 seed 平均；仅当 95% percentile CI 下界高于 −0.02 才判非劣。逐 seed delta/McNemar 只作透明报告，不再用主观 override。
