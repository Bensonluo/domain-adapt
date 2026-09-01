# Phase 1 证据清单

这是阶段级索引，不替代各 run 的配置和 hash。`PARTIAL` 表示产物存在但缺少完整 lineage 或研究有效性。

机器可读注册表：[`artifact_registry.json`](artifact_registry.json)。

评估隔离登记：[`audit/benchmark_registry.json`](audit/benchmark_registry.json)；协议见 [`audit/blind-test-protocol.md`](audit/blind-test-protocol.md)。

| 模块 | 主要产物 | 当前等级 | 缺口 |
|---|---|---|---|
| CPT 数据与配置 | `week10/`、`results/week12_*` | PRESENT | 1B–3B 目标未完成；早期 seed/model revision 不完整 |
| CPT 配比 | `results/week12_lora_cpt/sweep_summary.json` | PARTIAL | 单 seed；50/50 由复用 dev 选出 |
| 偏好数据 | `week14/pref_qc_report.json`、`audit/preference_split_audit.json` | PARTIAL | 新 grouped split 已实现且 prompt overlap=0；历史 split 有 50 组泄漏，质量仍无独立验证 |
| DPO beta sweep | `results/week15_dpo/sweep_summary.json` | PRESENT | matched n=13；无多 seed；无匹配 SFT baseline |
| DPO failure modes / IPO | `results/week16_*` | PRESENT | 目标同构评价、单 seed、小 matched bucket |
| GRPO | `results/week17_grpo/grpo_summary.json`、`audit/week17_clean_reanalysis.json` | INVALIDATED_PART | 8/500 eval 题进入 GRPO train；历史只存 50 条预测，完整 clean effect 仅可界定、不可点识别 |
| Response distillation | `results/week19_distill/distill_summary.json` | PRESENT | 单 seed；机制归因混杂 |
| Logit KD / rejection sampling | `results/week20_distill/week20_summary.json`、lineage | REPRODUCIBLE_PART | lineage 较完整；多臂单 seed、评测复用 |
| Synthetic replacement | `results/week21_synthetic/week21_summary.json`、lineage | REPRODUCIBLE_PART | 历史非劣效未成立；旧 RNG；无临床人工审核 |
| Synthetic clean confirmation data | `audit/week21_clean_replacement_audit.json` | READY_FOR_CONFIRMATION_RUN | 两臂 hash/标签/token 预算/零重叠已验证；尚未多 seed 重训 |
| Core confirmation specifications | `confirmation/*.json`、`confirmation/validate_specs.py` | FROZEN_BEFORE_EXECUTION | 五类合同通过机器校验；实验尚未运行，外部 blind 不可用 |
| Local remediation completion review | `audit/completion-review.md`、`audit/run_all.sh` | LOCALLY_VERIFIED | 25 项单测及三类 validator 通过；研究退出条件仍未满足 |

## 核心链路 schema

每个确认 run 应能追到：

`claim_id → run_id → data_hash → base_model_revision → training_config → seed → checkpoint → per_example_predictions → aggregate_summary → allowed_wording`

缺任一关键字段时，可保留为学习产物，但不得自动升级为 `VERIFIED`。
