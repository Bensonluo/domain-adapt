# Phase 0 交付物清单

| 周 | 已交付材料 | 内容与范围 |
|---|---|---|
| W1 | `week1/*.py`、`results/loss_curve.png`、checkpoint/sample | attention、Transformer 和训练循环实现；lm-eval 基线未留存 |
| W2 | `week2/*.py`、`results/week2_*` | nanoGPT 训练、不同 temperature 样例和复盘 |
| W3 | `week3/read_notes.md`、两个 comparison、训练脚本 | HF 源码阅读与框架比较；full FT 资源实测尚未运行 |
| W4 | `week4/*`、`results/week4_*` | LoRA/QLoRA 笔记、实现、PEFT 对比及随机矩阵 SVD 演示 |
| W5 | 脚本、notebook、checklist | chat template、loss masking 和 SFT 训练目标实现；模型效果消融尚未运行 |
| W6 | 外部 `4bit-QLoRA-post-training/medical_entity` @ `9267c7c569eeb9f2b14d0a1cf0faa67c831d7126` | 实际交付调整为 Qwen 系列药品实体匹配案例，覆盖数据、训练与结构化评估 |
| W7 | 五份 `derivation_*.md` | attention、softmax+CE、LoRA/SVD、DPO 和 AdamW 推导及纠错 |
| W8 | `eval_report.md`、judge 代码、rubric 模板、知识图谱 | 完成结构化评估报告；开放式 judge 与人工评分部分为原型和设计材料 |

## 外部案例

- `medical_entity`：Week 6 的 Qwen 系列药品实体匹配实践。
- `master_data`：Week 8 的独立评估案例，采用 Gemma 26B 进行机构/产品匹配；代码版本同为 `9267c7c569eeb9f2b14d0a1cf0faa67c831d7126`。

两个案例的模型和任务不同，分别记录其训练与评估结果。
