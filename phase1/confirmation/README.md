# Phase 1 确认实验冻结规范

这里的 JSON 是“运行前合同”，不是已完成实验。它们将探索阶段的候选方案收窄为一次性确认设计，并把最容易复发的方法学错误变成机器可拒绝的条件。

```bash
python3 phase1/confirmation/validate_specs.py
python3 -m unittest discover -s phase1/confirmation/tests -v
```

共同规则：至少 3 个独立训练 seed；固定 control/treatment、数据 hash 和主指标；CMExam 6,305 题仅是本地确认候选，不是 blind test；任何正式阶段退出仍要求外部医学和通用 blind evaluation。执行后若根据确认结果调参，必须创建新 claim，不能覆盖本规范或把同一候选重复当作确认集。

五个规范分别覆盖 CPT、DPO/IPO、GRPO、hard-vs-soft KD 和 synthetic replacement。`FROZEN_BEFORE_EXECUTION` 表示方案已锁定但尚未运行，不能误读为结论已验证。
