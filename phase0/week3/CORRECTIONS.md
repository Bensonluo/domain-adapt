# Week 3 纠错记录

> 审查日期：2026-08-28｜状态：`PARTIAL`

## C1：有全量微调脚本，但没有全量微调实测

- 原问题：周目标描述为“做过全量微调并亲身看到显存成本”，实际只有 `train_full_ft.py`，缺少日志、显存峰值、硬件、训练时间和 loss 曲线。
- 为什么错：脚本可运行性不能替代实验执行，也不能支持资源成本结论。
- 已修复：Week 3 和阶段总结均标记为“源码阅读/脚本准备存在，全量微调实测未完成”。
- 关闭条件：在可负担小模型上跑一次 full FT，记录硬件、精度、优化器、峰值显存、吞吐和 loss；或正式标记 `WAIVED`。

## C2：DataCollator 对 causal LM shift 的表述不准确

- 原问题：自测答案写成 collator 创建“右移一位”的 labels。
- 为什么错：常见 causal LM 流程是 collator 复制/整理 labels，模型内部在计算 loss 时完成 logits/labels shift。
- 修复方法：README 改为区分 padding/masking、labels 构造和模型内部 shift。

## C3：24GB 只是参数与优化器的简化下界

- 原问题：`4 × 参数量 × 4 bytes` 容易被理解为完整训练显存。
- 为什么错：实际还受 mixed precision master weights、激活、临时 buffer、梯度 checkpointing 和实现影响。
- 修复方法：保留 FP32 AdamW 约 24GB 的教学估算，同时明确它不含激活和框架开销。

## C4：full FT 脚本随机按行拆分并把 dev 叫 test

- 原问题：训练脚本在数据加载后直接 `train_test_split(0.1)`，没有分组规则、seed 或独立 dev 输入。
- 为什么错：同 prompt、实体、来源或模板可能跨集合；同时训练过程反复查看的集合不是最终 test。
- 已修复：强制传入预先分组切分的 `--eval_data`，每个 epoch 评估并按 `eval_loss` 恢复最佳 checkpoint，显式记录 seed。最终 blind test 保持独立。
