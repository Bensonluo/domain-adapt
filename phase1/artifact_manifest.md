# Phase 1 交付物清单

本清单汇总已经完成的材料、实验及后续可继续研究的问题。各次运行的配置、结果和数据来源可从对应路径查看。

机器可读索引：[`artifact_registry.json`](artifact_registry.json)。
评估数据记录：[`audit/benchmark_registry.json`](audit/benchmark_registry.json)；使用背景见 [评估数据与后续验证](audit/blind-test-protocol.md)。

| 模块 | 已交付材料 | 结果与后续研究方向 |
|---|---|---|
| CPT 数据与配置 | `week10/`、`results/week12_*`：数据处理、训练配置和真实语料实验 | 实际采用约 1414 万 token；原 1B–3B token 扩展未执行 |
| CPT 配比 | `results/week12_lora_cpt/sweep_summary.json`：三比例对比 | 单 seed 下观察到正 gain；50/50 用作后续基线，比例差异的稳定性可进一步测试 |
| 偏好数据 | `week14/pref_qc_report.json`、`audit/preference_split_audit.json`：质量分析与重新分组切分 | 历史 split 有 50 个 prompt 组交叉；新 grouped split 的 prompt overlap=0 |
| DPO beta sweep | `results/week15_dpo/sweep_summary.json`：三组 beta 结果 | matched bucket 为 13 条，差异较小；更大样本与匹配 SFT baseline 有助于比较 |
| DPO failure modes / IPO | `results/week16_*`：失败模式与 IPO 对比 | 发现长度偏差和目标相关的指标差异；独立任务指标有助于判断泛化 |
| GRPO | `results/week17_grpo/grpo_summary.json`、`audit/week17_clean_reanalysis.json`：训练与历史重分析 | 8/500 题与训练重叠；因只保存 50 条逐题预测，完整 clean 效应可界定范围、无法恢复精确值 |
| Response distillation | `results/week19_distill/distill_summary.json`：三臂对比 | teacher explanation 出现积极信号；长度、风格等因素可分别研究 |
| Logit KD / rejection sampling | `results/week20_distill/week20_summary.json`、lineage：多臂实验与来源记录 | 已形成候选配置和方法比较；重复 seed 可帮助估计波动 |
| Synthetic replacement | `results/week21_synthetic/week21_summary.json`、lineage：生成、质检和替代实验 | 点估计 −0.8pp；区间尚不支持 −2pp margin 下的非劣效，抽检为非临床 AI 复核 |
| Synthetic clean confirmation data | `audit/week21_clean_replacement_audit.json`：两臂清理后的训练数据 | hash、标签、token 预算和零重叠已检查；新数据尚未重训 |
| 后续实验设计 | `confirmation/*.json`、`confirmation/validate_specs.py` | 五类设计与配置检查已完成，实验尚未执行 |
| 修订与验证记录 | `audit/completion-review.md`、`audit/run_all.sh` | 已记录 25 项单测及三类 validator 的通过结果 |

## 实验追溯

索引关联问题、运行、数据版本、模型、配置、seed、checkpoint、逐题预测和汇总结果，方便解释结果或复现实验。历史运行中缺少的信息在相应条目中说明，例如 Week 17 缺少完整逐题预测，因此无法重算精确配对区间。
