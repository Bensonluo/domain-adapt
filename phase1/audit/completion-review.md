# Phase 1 本地整改完成审查

日期：2026-08-29

## 审查范围

本记录只判断 `.omc/prd.json` 中 P1R-001～P1R-005 的“本地可解决整改”是否完成，不把尚未执行的多 seed 训练或尚不存在的外部 blind evaluation 写成完成。阶段状态继续为 `REMEDIATION_REQUIRED`。

## 独立审查与处置

独立只读审查先后指出并复验了以下问题：确认 validator 的 path/hash 与路径别名绕过；CPT exposure/adaptation-mode 混为一个 estimand；起始权重只锁 config；GRPO probes 未冻结；多 seed 非劣效合并规则不明确；文档/registry 小范围漂移。最终 P1R-001、P1R-002、P1R-003、P1R-004 均获独立审查批准。

修复后：

- 五类合同固定 immutable revision、真实起始 `model.safetensors` hash/size、训练 recipe、3 个 seed、确认候选和统计规则。
- CPT 拆成 `cpt_exposure` 与 `adaptation_mode` 两个 estimand。
- GRPO 使用 801 条实际 probe fixture，并锁定 generator/fixture/audit 三个 hash、固定选择算法和数值阈值。
- synthetic 非劣效采用 seed × paired-item 两层 bootstrap，CI 下界判定固定为 `> -0.02`。
- validator 对缺失 hash、旧 holdout 路径别名、错误 blind/盲测措辞、config 冒充 weights、空 probe 阈值等 fail closed。

## bounded cleanup

`ai-slop-cleaner` 仅作用于本轮 Phase 1 新增/修改文件，执行了以下保行为清理：

- 删除 Week 21 测试死变量；
- 复用已有 `compact_text`，删除自造的重复题目键逻辑；
- 避免 Week 17 重分析循环中重复构造集合；
- 压低 `run_all.sh` 数据生成步骤的冗长输出；
- 保留所有审计数值、hash、状态和因果边界。

清理未以 Phase 0 或 `phase1/.zcode` 为目标，也未修改其中任何文件。工作树中既有 Phase 0 变更和用户自有 `.zcode` 内容均按原样保留。

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

## 最终边界

本地设计、历史纠错、确认合同和审计链已闭环；研究结论尚未升级。外部医学/通用 blind evaluation、label custodian、五类多 seed 确认训练以及合成数据临床人工盲评仍是正式退出条件。
