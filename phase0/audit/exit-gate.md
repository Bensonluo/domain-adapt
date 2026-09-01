# Phase 0 退出门槛

## 状态定义

状态按“执行、证据、有效性、处置”四类使用；组合标签（如 `EXTERNAL_PARTIAL`、`INVALIDATED_PART`）表示同时满足对应定义，不是新的通过等级。

- `DONE`：执行动作完成。
- `PRESENT`：有可定位产物，但未必可复现。
- `REPRODUCIBLE`：数据、配置、代码、日志和结果足以重跑。
- `PARTIAL`：仅满足部分目标或存在范围限制。
- `EXTERNAL`：证据位于其他仓库或系统；必须固定 commit/版本与运行 manifest 才可参与验收。
- `UNVERIFIED`：声明尚无足以判定真假的受控证据。
- `VERIFIED`：证据支持对应声明，且主要混杂已控制。
- `INVALIDATED`：当前证据与声明矛盾或实验设计不能回答该问题。
- `INVALIDATED_CAUSAL`：观测数字可保留，但因混杂不能归因给声明中的原因。
- `INVALIDATED_PART`：复合交付中至少一个核心子声明无效；其余部分必须分别标状态。
- `RETRACTED`：旧声明已从当前总结和决策中永久撤回；错误历史保留，但不再作为待证明声明。
- `REPLACED`：旧声明已由范围更窄、证据可映射的新声明取代；必须链接新 claim ID。
- `WAIVED`：明确放弃并记录理由，不再计入退出标准。

状态迁移规则：`UNVERIFIED → VERIFIED/INVALIDATED/WAIVED`；`INVALIDATED → RETRACTED`，或在新证据与新 claim ID 下标记 `REPLACED`。不得删除错误行来“通过”验收，也不得把旧声明直接从 `INVALIDATED` 改成 `VERIFIED` 而不保留原证据和替代声明。

## 通过条件

Phase 0 只有同时满足下列条件才能从 `REMEDIATION_REQUIRED` 改为 `PASSED`：

1. 必选产物都有稳定路径；外部证据固定到 commit、数据版本、配置和结果文件。
2. Week 3 全量微调资源实验已完成，或正式标记 `WAIVED` 并修改阶段目标。
3. masking 效果与数据质量/数量至少完成最小受控实验。
4. LoRA/SVD 不再引用随机矩阵作为真实 ΔW 证据；如保留经验数字，必须来自真实 full-FT 更新。
5. base 与 finetuned 使用同 runtime、量化、prompt、模板、解码和完整同一评估集重跑。
6. 当前反复使用的评估集降级为 dev；另有冻结后只运行一次的 blind test。
7. 结构化任务保存逐题预测并报告 paired CI/McNemar；至少 3 个训练 seed。
8. 若阶段仍声称覆盖开放式能力，则完成 blind human evaluation、至少两名评分者和分维度 IAA；LLM judge 只作为补充。
9. Phase 0 总结中的每个结果性声明都能映射到 `claim-evidence-matrix.md`；所有仍在使用的核心声明均为 `VERIFIED` 或明确 `WAIVED`。历史错误必须保留为 `RETRACTED/REPLACED`，但不阻塞退出。

## 当前判定

`REMEDIATION_REQUIRED`。学习性主体完成，但研究性验收未通过。
