# Week 20 纠错记录

## C20-01：logit KD 不等价于所有 feature distillation

- 原问题：将跨框架无法方便做 hidden-state matching 改写为“现代生成式 LLM feature distill 等价于 logit KD”。
- 为什么错：这是本项目实现约束下的选择，不是方法定义上的等价。
- 修复：命名为 `logit-level knowledge distillation`；hidden-state feature distillation 仅标为本轮未实现。

## C20-02：从单 seed 多臂 winner 推出普适规律

- 原问题：“soft label 保知识、hard label 毁知识”“纯 KL 最佳、丢掉 hard CE”由同一 dev 上六臂点估计得出。
- 为什么错：存在多重比较、winner's curse，hard/soft 条件也不是完全同构控制。
- 修复：`kd_pure` 只称开发集点估计 winner；机制结论标 `UNVERIFIED`，按统一数据/token budget、多 seed 重跑。

## C20-03：proxy reward 失败被过早称为 reward hacking

- 原问题：`rs_teacher` 仅比 base 少 1/500，便归因 teacher judge 反噬。
- 修复：teacher-only 选择在本次运行未改善目标任务，proxy mismatch 是一种可能解释；一题差异也可能来自运行波动，尚不足以归因于 reward hacking。

## C20-04：on-policy 范围不准确

- 原问题：一轮 best-of-N + rejection-sampling SFT 被外推为一般 on-policy distillation。
- 修复：当前方法统一称“一轮 student-sampled rejection-sampling SFT”；不外推到迭代 GKD、RL 或完整 on-policy 家族。

## C20-05：蒸馏确认比较已预注册

- 修复：`phase1/confirmation/distillation.json` 只比较预先锁定的 `alpha=[1,0.5,0]`，主比较是 soft KL vs hard CE，使用 paired bootstrap、McNemar 和 Holm 校正。规范已通过校验但尚未运行，不能把它写成机制结论。
