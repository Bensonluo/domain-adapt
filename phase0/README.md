# Phase 0：基础实现与领域微调实践

> 8 周内容涵盖 Transformer、训练循环、LoRA/QLoRA、SFT 和评估，形成代码、推导笔记、实验报告及结构化匹配案例。

## 阶段成果

- 从基础模块实现走到领域 SFT，记录训练目标、数据处理和工程选择。
- 在同源合成机构匹配评估中，微调模型及当时推理链路的 Top-1 为 98.75%，基座及另一推理链路为 79.75%。
- 复盘中识别出 runtime、解码、数据分布等对比较的影响，区分实际观察和原因解释。
- Week 6 的 `medical_entity` 与 Week 8 的 `master_data` 分别展示训练实践和独立评估案例。

## 每周交付

| 周 | 交付内容 | 进一步可研究的问题 |
|---|---|---|
| Week 1 | autograd、attention、Transformer 和 toy 训练 | 补充 lm-eval 基线与测试输出 |
| Week 2 | nanoGPT 训练、生成样例和复盘 | 比较采样设置对生成结果的影响 |
| Week 3 | HF 源码阅读、框架比较和 full-FT 脚本 | 实测全量微调显存与耗时 |
| Week 4 | LoRA/QLoRA 实现、论文笔记和 SVD 教学演示 | 真实参数更新的谱分布与 rank 选择 |
| Week 5 | template、masking 和 SFT 训练脚本 | 比较不同 loss 目标与数据构成 |
| Week 6 | 外部药品实体匹配案例 | 按业务需要扩展任务或样本 |
| Week 7 | 五份数学推导与 SVD 纠错 | 将推导与具体训练现象联系起来 |
| Week 8 | 结构化评估报告、judge 原型、rubric 和知识图谱 | 同条件模型比较与新分布评估 |

## 结果与复盘

- [阶段总结](notes/phase0_summary.md)
- [交付物清单](artifact_manifest.md)
- [结果与证据对照](audit/claim-evidence-matrix.md)
- [结果解读与改进方向](audit/exit-gate.md)
- [后续实验建议](audit/rerun-plan.md)
- [修订思路](audit/ADR-001-evidence-first-remediation.md)

每周的 `CORRECTIONS.md` 记录问题、原因、修改和后续思考。
