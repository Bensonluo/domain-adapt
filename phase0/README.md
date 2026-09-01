# Phase 0 状态与证据入口

> 状态：`REMEDIATION_REQUIRED`
>
> 准确表述：基础学习与工程练习大部分完成；结构化领域 SFT POC 已完成；研究性消融、开放域评估和正式退出验收尚未完成。

本文件是 Phase 0 的唯一阶段状态源。各周 README 记录执行内容，但不得单独把阶段标记为通过。

## 当前边界

- 可以声称：在同源合成的机构匹配评估上，“微调模型 + 当时推理链路”观察到明显提升。
- 不能声称：提升完全由 LoRA/SFT 导致、能迁移到真实医疗业务、稳定超越所有大模型/API、延迟下降由微调导致。
- Week 6 的 `medical_entity` 与 Week 8 的 `master_data` 是两个独立案例，不构成同一模型从训练到评估的闭环。

## 周状态

| 周 | 执行 | 证据 | 有效性 | 当前判定 |
|---|---|---|---|---|
| Week 1 | DONE | PARTIAL | PARTIAL | 核心代码存在，基线与行为验收缺失 |
| Week 2 | DONE | PRESENT | PARTIAL | 训练与样例存在，命名/验收未同步 |
| Week 3 | PARTIAL | PARTIAL | PARTIAL | HF 阅读存在，全量微调实测缺失 |
| Week 4 | DONE | PARTIAL | INVALIDATED_PART | LoRA 实现存在，SVD 实证主张撤回 |
| Week 5 | PARTIAL | PARTIAL | INVALIDATED_PART | 实现存在，masking 效果和质量/数量结论未验证 |
| Week 6 | EXTERNAL | PARTIAL | PARTIAL | 外部结构化匹配案例，非原计划完整替代 |
| Week 7 | DONE | PRESENT | PARTIAL | 五份推导存在，能力性复述未验收，SVD 部分已纠错 |
| Week 8 | PARTIAL | PARTIAL | PARTIAL | 结构化评估存在，开放式 judge/人工 IAA 未闭环 |

状态字段含义见 [退出门槛](audit/exit-gate.md)。

## 审查与修复资料

- [架构决策：证据优先修复](audit/ADR-001-evidence-first-remediation.md)
- [声明—证据矩阵](audit/claim-evidence-matrix.md)
- [Phase 0 退出门槛](audit/exit-gate.md)
- [最小重跑计划](audit/rerun-plan.md)
- [产物清单](artifact_manifest.md)

每周错误原因和修复历史保存在对应目录的 `CORRECTIONS.md`。
