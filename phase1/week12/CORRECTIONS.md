# Week 12 纠错记录

## C12-01：把多变量变化归因给 LoRA

- 原问题：从旧 Qwen3.5-0.8B、MLX full 模式、200 iter 切到 Qwen3-1.7B、LoRA、2500 iter 后，写成“LoRA 直接验证了无遗忘/使 gain 转正”。
- 为什么错：模型、架构、训练方式、步数、tokenizer/数据规模同时改变，无法识别 LoRA 的独立因果效应。
- 修复：只保留“新配置上观察到正 gain，所测 dev 子集未下降”；LoRA 因果结论标 `UNVERIFIED`。

## C12-02：50/50 被称为最优比例

- 原问题：三比例单 seed 点估计差约 1pp，却选择 50/50 并在后续称最优。
- 为什么错：差异在噪声量级，且同一 CMMLU dev 被反复选优，存在 winner's curse。
- 修复：50/50 仅称“后续采用的操作性基线”；重复 seed 与新的评估样本有助于判断比例差异是否稳定。

## C12-03：“灾难性遗忘 = 0”范围过宽

- 原问题：四个 CMMLU 子集略涨被外推为没有遗忘。
- 修复：改为“本次运行在所测子集上未观察到下降”；不能外推到未测能力。

## C12-04：后续实验的比较条件不够具体

- 为什么错：原计划没有明确记录 seed、数据 hash、唯一处理变量和多重比较策略，重跑时仍可能同时改变模型、预算与训练方式。
- 修复：新增 `phase1/confirmation/cpt.json`，固定 no-CPT/LoRA/full 三臂、3 个 seed、同一训练数据和优化 token 预算；full 臂运行前必须导出可训练参数名。规范状态是“已冻结、未执行”，不代表 CPT 结论已验证。

## C12-05：把 no-CPT 与 LoRA/full 混成一个处理变量

- 原问题：首版确认合同把 no-CPT、LoRA-CPT、full-CPT 放在一个三臂比较里，却称唯一处理变量是 adaptation mode；no-CPT 实际还改变了 CPT exposure 和优化预算。
- 修复：拆成两个 estimand。A 只检验 no-CPT vs LoRA-CPT 的 exposure 效应，明确优化 token 不可能固定；B 在相同数据与 token 预算下比较 LoRA-CPT vs full-CPT 的 adaptation-mode 效应。两者分别报告并做 Holm 校正，不互相替代。
