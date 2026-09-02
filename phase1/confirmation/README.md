# Phase 1 后续实验设计

本目录保存 CPT、DPO/IPO、GRPO、hard-vs-soft KD 和 synthetic replacement 的五类实验设计。配置已经整理并检查，实验尚未执行。

这些设计采用 3 个独立训练 seed，固定对照组、数据版本和主指标，以进一步比较收益来源与运行波动。CMExam 6,305 题是本地确认候选，标签可见，不是外部盲测。若后续根据结果调整方案，可另存配置版本，保留探索过程。

```bash
python3 phase1/confirmation/validate_specs.py
python3 -m unittest discover -s phase1/confirmation/tests -v
```

`FROZEN_BEFORE_EXECUTION` 表示当前设计版本已记录、尚未运行。配置检查用于发现路径、hash、对照设置等问题，不评价项目是否完成。
