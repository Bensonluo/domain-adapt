# Phase 1 方法论修订与验证记录

日期：2026-08-29

## 审查范围

本记录汇总数据划分、历史结果重分析、确认实验规范和验证脚本的修订。多 seed 训练与外部 blind evaluation 尚未执行，与本次已完成的修订分开记录。

## 问题与修复

修订前的问题包括：确认 validator 的 path/hash 与路径别名校验不完整；CPT exposure/adaptation-mode 混为一个 estimand；起始权重只锁 config；GRPO probes 未冻结；多 seed 非劣效合并规则不明确；文档与 registry 不一致。

修复后：

- 五类实验设计固定 immutable revision、真实起始 `model.safetensors` hash/size、训练 recipe、3 个 seed、确认候选和统计规则。
- CPT 拆成 `cpt_exposure` 与 `adaptation_mode` 两个 estimand。
- GRPO 使用 801 条实际 probe fixture，并锁定 generator/fixture/audit 三个 hash、固定选择算法和数值阈值。
- synthetic 非劣效采用 seed × paired-item 两层 bootstrap，CI 下界判定固定为 `> -0.02`。
- validator 对缺失 hash、旧 holdout 路径别名、错误 blind/盲测措辞、config 冒充 weights、空 probe 阈值等 fail closed。

## 代码整理

同步整理了验证脚本，保持审计结果和行为不变：

- 删除 Week 21 测试死变量；
- 复用已有 `compact_text`，删除自造的重复题目键逻辑；
- 避免 Week 17 重分析循环中重复构造集合；
- 压低 `run_all.sh` 数据生成步骤的冗长输出；
- 保留所有审计数值、hash、状态和因果边界。

## 清理后完整回归

清理及审查修复完成后执行：

```bash
bash phase1/audit/run_all.sh
```

结果：exit code 0。

- Week 17 重分析：3/3 tests passed；
- confirmation contracts：17/17 tests passed；
- Week 21：5/5 tests passed；
- evidence protocol invariants：PASS；
- 5 个 frozen confirmation specifications：PASS；
- repository consistency：14 JSON、146 local links、17 registered artifact references，PASS。

合计 25 项单测通过。该回归会重新生成 preference split 审计、benchmark 审计、6,305 题确认候选、Week 17 重分析、GRPO probe fixture 和 Week 21 clean matched 两臂，随后再检查 hash 与不变量。

## 完成内容与后续选择

已完成本地设计、历史纠错、后续实验配置和验证脚本。后续可围绕一个具体问题开展多 seed 比较或外部样本评估；若研究临床应用，再安排专业评审。这些是继续研究的选项，不影响本次修订和既有实验的交付记录。
