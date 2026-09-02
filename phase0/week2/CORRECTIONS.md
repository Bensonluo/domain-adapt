# Week 2 纠错记录

> 复盘日期：2026-08-28

## C1：总结周次映射错误

- 原问题：Phase 0 总结把 nanoGPT 写成 Week 3，实际 nanoGPT 是 Week 2。
- 为什么错：这会掩盖真正的 Week 3 HF/full-FT 缺口。
- 已修复：总结恢复为 Week 2 nanoGPT、Week 3 HF/full FT 两个独立阶段。

## C2：产物路径与计划漂移

- 原问题：计划要求统一训练记录和样例分析，实际保存为多个 temperature 样例和 `week2_nanogpt_takeaways.md`。
- 为什么错：不是实验错误，但会让自动验收误判缺失，也使总结难以追溯。
- 已修复：在 `phase0/artifact_manifest.md` 建立计划产物到实际产物映射，直接引用已有文件。
- 后续建议：需要复现时，可从现有记录汇总数据、配置、seed 和 loss，无需为了文件名重复制作材料。
