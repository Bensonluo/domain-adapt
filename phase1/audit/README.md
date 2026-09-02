# Phase 1 方法论审计入口

## 一键重建与验证

```bash
bash phase1/audit/run_all.sh
```

等价的展开命令如下，便于定位某一步失败：

```bash
python3 phase1/audit/build_preference_grouped_split.py
python3 phase1/audit/audit_benchmark_splits.py
python3 phase1/audit/build_cmexam_confirmation_candidate.py
python3 phase1/audit/reanalyse_week17_clean.py
python3 phase1/confirmation/build_grpo_probe_fixtures.py
phase1/.venv/bin/python phase1/week21/prepare_clean_matched_replacement.py \
  --real-raw phase1/data/processed/cmexam/train.jsonl \
  --synthetic phase1/results/week21_synthetic/data/evol_instruct.jsonl phase1/results/week21_synthetic/data/self_instruct.jsonl \
  --confirmation phase1/data/processed/cmexam/confirmation_candidate_v1.jsonl \
  --control-output phase1/data/processed/confirmation/week21_control_real_clean_v1.jsonl \
  --treatment-output phase1/data/processed/confirmation/week21_treatment_synthetic50_clean_v1.jsonl \
  --audit-output phase1/audit/week21_clean_replacement_audit.json \
  --n-total 2000 --synthetic-fraction 0.5 --real-pool-size 4000 --similarity-threshold 0.78 \
  --tokenizer phase1/results/week12_lora_cpt/50_50_fused --seed 123
python3 -m unittest discover -s phase1/audit/tests -v
python3 -m unittest discover -s phase1/confirmation/tests -v
python3 -m unittest discover -s phase1/week21/tests -v
python3 phase1/audit/validate_evidence_protocol.py
python3 phase1/confirmation/validate_specs.py
python3 phase1/audit/validate_repository_consistency.py
```

前半段命令生成或刷新审计产物，后半段只读验证 hash、污染状态、确认合同和关键不变量。任一步失败，整套入口返回非零状态。

`confirmation/validate_specs.py` 检查五类确认合同的 seed、hash、对照变量、统计方案和 blind-test 边界；规范说明见 [`../confirmation/README.md`](../confirmation/README.md)。

后续实验可使用 [`confirmation_run_manifest.template.json`](confirmation_run_manifest.template.json) 记录配置。若根据结果调整方案，另存版本有助于区分探索过程与原始设计。

## 已完成的分析与修订

- 历史 preference random split 的泄漏已量化：50 个 normalized prompt 组交叉。
- 新 grouped v1 split 已生成：1299 train / 100 dev，prompt-group overlap=0。
- CMExam/CMMLU 的 development 污染状态和 hash 已登记。
- CMExam historical 500 中 8 题进入 GRPO 8K train 的直接泄漏已记录。
- Week 17 已完成可恢复重分析：完整 492 clean 题只能得到 +0.61pp 至 +3.86pp 边界；因历史只保留 50 条预测，精确 clean 统计不可恢复。
- 6,305 题 CMExam clean confirmation candidate 已冻结，和已知 Phase 1 训练 prompt 的规范化交叉为 0。
- Week 21 clean confirmation 两臂已生成：各 2,000 条、同标签分布、token 预算差不超过 1%、对确认候选 prohibited overlap=0。

## 尚未开展的后续实验

- 6,305 题确认集不是外部 blind test。
- 外部医学和通用样本评估尚未配置。
- Week 21 clean 数据已就绪，但多 seed control/treatment 尚未重训；历史结果仍对应含 12 条 overlap 的数据。
- CPT、DPO/IPO、GRPO、蒸馏和合成数据的多 seed 确认实验尚未执行。

具体结果解读及实验选择见 [分析与改进方向](exit-gate.md)。
