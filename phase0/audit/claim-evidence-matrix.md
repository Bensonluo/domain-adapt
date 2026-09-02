# Phase 0 声明—证据矩阵

| ID | 声明 | 当前证据 | 主要问题 | 状态 | 修订后的说明 |
|---|---|---|---|---|---|
| P0-C01 | 已完成 HF 全量微调并理解显存成本 | `week3/train_full_ft.py` | 无日志、峰值显存、loss 曲线 | UNVERIFIED | 已准备脚本并完成部分源码阅读 |
| P0-C02 | Qwen ΔW 的 rank 8 捕获 85%–95% 能量 | 随机矩阵 notebook | 不是 Qwen/真实 ΔW；能量公式曾不一致 | RETRACTED | notebook 仅演示 SVD 概念 |
| P0-C03 | rank=8 通常足够 | 理论笔记 | 无本项目受控 rank 消融 | RETRACTED | rank 是需按任务验证的容量超参数 |
| P0-C04 | masking 优于不 masking | token 标签展示 | 没有训练效果对照 | UNVERIFIED | assistant-only loss 使目标更贴近期望输出 |
| P0-C05 | 数据质量大于数量 | 计划中的 500/2000/5000 三组 | 同时改变质量、数量和 token budget；无报告 | RETRACTED | 数据质量各维度的效应需分别验证 |
| P0-C06 | Week 6 完成原定开放式领域 SFT 闭环 | 外部 `medical_entity` 案例 | 模型、数据、任务、指标与原计划不同；人工评估缺失 | PARTIAL | 完成了结构化实体匹配替代案例 |
| P0-C07 | 26B finetuned 的 +19pp 由 SFT 导致 | 外部 `master_data` 报告 | base/FT runtime、解码配置混杂；测试集反复使用 | PARTIAL | 在当时两条完整链路间观察到 +19pp |
| P0-C08 | 微调使延迟降低 64% | 不同 runtime 的延迟 | MLX 与 LM Studio/API 不可归因 | RETRACTED | 只报告各自链路观测延迟 |
| P0-C09 | 微调模型稳定超越更大模型和商业 API | 跨模型排行榜 | 样本量、后端、提示和解码不完全一致 | RETRACTED | 排行榜为描述性历史记录 |
| P0-C10 | rank=32 对匹配任务必要 | 当前最终配置 | 无匹配条件的 r=8/16/32 消融 | RETRACTED | 当前配置选用了 rank=32 |
| P0-C11 | 已建立 benchmark + judge + 人工评估三层体系 | 结构化报告、judge 代码、空 rubric | judge 未运行，人工评分和 IAA 缺失 | PARTIAL | 已完成结构化自动评估，开放式评估待补 |
| P0-C12 | 98.75% 可代表真实业务表现 | 同源合成 holdout | 外部效度不足，无真实盲测 | RETRACTED | 98.75% 仅适用于该合成评估分布 |

本表保留早期结论与修订依据，便于查阅数值来源、理解解释为何变化。状态描述具体结论的证据情况，不是对已交付材料的评级。
