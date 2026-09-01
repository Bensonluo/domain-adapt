# Phase 0 产物清单

| 周 | 计划产物 | 实际证据 | 状态 | 备注 |
|---|---|---|---|---|
| W1 | attention/transformer/训练/lm-eval 基线 | `week1/*.py`、`results/loss_curve.png`、checkpoint/sample | PARTIAL | lm-eval 与行为验收缺失 |
| W2 | nanoGPT 训练、样例、复盘 | `week2/*.py`、`results/week2_*` | PRESENT | 实际文件名与计划不同 |
| W3 | HF 笔记、full FT 日志/显存/loss | `week3/read_notes.md`、两个 comparison、训练脚本 | PARTIAL | full FT 实测结果缺失 |
| W4 | LoRA/QLoRA 笔记、实现、对比、SVD | `week4/*`、`results/week4_*` | PARTIAL | notebook 仅为随机矩阵教学演示 |
| W5 | template、masking 对照、质量/数量报告 | 脚本、notebook、checklist | PARTIAL | 两个训练对照均缺失 |
| W6 | 数据、模型、loss、人工评估 | 外部 `4bit-QLoRA-post-training/medical_entity` @ `9267c7c569eeb9f2b14d0a1cf0faa67c831d7126` | EXTERNAL_PARTIAL | 代码证据已固定 commit；仍需补运行/数据/checkpoint manifest 与人工评估 |
| W7 | 五份推导、复习/白板证据 | 五份 `derivation_*.md` | PARTIAL | 内容存在，行为验收缺失；SVD 已纠错 |
| W8 | benchmark、judge、人工 IAA、总结 | `eval_report.md`、judge 代码、rubric 模板、知识图谱 | PARTIAL | 结构化评估存在，开放式评估未完成 |

外部案例必须拆分记录：

- `medical_entity`：Week 6 替代性交付，Qwen 系列、药品实体匹配。
- `master_data`：Week 8 独立评估案例，Gemma 26B、机构/产品匹配；代码证据同样固定到 `9267c7c569eeb9f2b14d0a1cf0faa67c831d7126`。

二者不得合并成同一训练—评估闭环。
