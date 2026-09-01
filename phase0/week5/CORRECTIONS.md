# Week 5 纠错记录

> 审查日期：2026-08-28｜状态：`PARTIAL / TWO CLAIMS UNVERIFIED`

## C1：masking 功能演示被当成效果对照

- 原问题：脚本只展示参与 loss 的 token 数，README 却勾选“不 masking vs masking 效果实验完成”。
- 为什么错：没有训练 A/B 模型，就不能证明下游正确率、prompt echo 或格式遵循发生改善。
- 已修复：验收拆成“实现/标签检查已完成”和“训练效果对照未完成”。
- 关闭条件：按 `phase0/audit/rerun-plan.md` 的 E2 运行三组同条件实验。

## C2：assistant 结束 token 被错误 mask

- 原问题：checklist 和 `loss_masking.py` 将 `<|im_end|>` 排除在监督之外。
- 为什么错：turn-ending/EOS 是目标回答的一部分；不监督它会削弱模型学习何时停止和如何闭合对话格式。
- 已修复：assistant content 和对应 `<|im_end|>` 均参与 loss；prompt、角色前缀和 padding 才设为 `-100`。

## C3：marker 找不到时 fail-open

- 原问题：训练脚本找不到 assistant marker 时退化为全序列 loss。
- 为什么错：数据/template 错配会被静默掩盖，实验名义上是 assistant-only，实际训练目标却变化。
- 已修复：改为抛出错误并停止，让数据问题显式暴露。

## C4：“质量 > 数量”实验混杂

- 原问题：500 高质量、2000 中质量、5000 低质量同时改变质量、数量、token 和训练更新量，并且没有结果报告。
- 为什么错：无法识别任何单一因素的作用。
- 已修复：撤销完成状态和确定性结论；设计固定数量的质量对照与固定质量的数量曲线。

## C5：经验超参被写成普适规则

- 原问题：固定 LR/epoch/rank，并写“target modules 加更多更好”。
- 为什么错：更多模块增加容量和成本，也可能增加过拟合；超参依赖任务与模型。
- 已修复：改成起始范围，并要求 dev 集、计算预算和消融共同决定。

## C6：传入 dev 集但训练期间没有评估

- 原问题：`sft_trainer.py` 接收 `eval_dataset`，但 `TrainingArguments` 没有启用 eval strategy，默认训练过程中不运行 dev 评估。
- 为什么错：无法监测泛化误差和过拟合，也无法证明最终 checkpoint 是按预设标准选择的。
- 已修复：每个 epoch 运行 dev loss，保存策略与评估对齐，按 `eval_loss` 恢复最佳 checkpoint，并显式记录随机 seed。

## C7：chat template 后可能重复添加 special tokens

- 原问题：先用 `apply_chat_template(..., tokenize=False)` 生成已含控制 token 的文本，再调用 tokenizer 时没有关闭 `add_special_tokens`。
- 为什么错：对会自动添加 BOS/EOS 的 tokenizer，可能重复边界 token，使训练格式偏离模板；是否触发又依赖具体模型，难以跨模型复现。
- 已修复：二次 tokenize 显式使用 `add_special_tokens=False`，边界只由 chat template 决定。

## C8：多轮样本截断后可能静默少监督一轮

- 原问题：若第一轮 assistant 完整、后续 assistant marker 在 512 token 截断点之后，代码仍因“至少找到一轮”而通过。
- 为什么错：名义上的多轮训练样本被静默改变，且丢失监督的比例随长度分布变化。
- 已修复：将原始 `messages` 中 assistant turn 数与 tokenized 后识别数逐样本比对；不一致立即报错，要求调整截断策略或 max length。
