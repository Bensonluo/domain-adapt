# Phase 1 声明—证据矩阵

| ID | 声明 | 当前证据 | 主要问题 | 状态 | 当前允许表述 |
|---|---|---|---|---|---|
| P1-C01 | 已完成 1B–3B token 严肃 CPT | Week 10 管线；约 1414 万 token 真实语料 | 与目标规模差两个数量级；早期检查项互相矛盾 | RETRACTED | CPT 数据管线和小规模真实语料实验已完成 |
| P1-C02 | LoRA 隔离领域学习并消除了灾难性遗忘 | Week 12 三个 LoRA-CPT 单次运行 | 相比旧 full 模式同时更换模型、方法、步数和数据处理；单 seed | UNVERIFIED | 在该次 LoRA-CPT 配置上，所测 CMMLU 子集未观察到下降 |
| P1-C03 | 50/50 是最优 CPT 混合比例 | Week 12 三比例点估计 | 差异约 1pp、单 seed、同一 dev 反复选优 | UNVERIFIED | 50/50 是后续实验采用的操作性基线，不代表已证实最优 |
| P1-C04 | DPO 没有造成灾难性遗忘 | Week 15 三个 beta 的 CMMLU 点估计 | 单 seed、小评测集且反复使用；历史 preference split 有 50 个 prompt 组交叉 | SUPPORTED_TREND | 该次 DPO sweep 未在所测开发子集上观察到明显下降 |
| P1-C05 | beta=0.1 是最优 DPO 强度 | Week 15 matched bucket 3/13 vs 2/13 | n=13、50 个 prompt 组泄漏，且文档自身认定为噪声 | RETRACTED | beta=0.1 只是历史脚本按点估计选出的候选 |
| P1-C06 | IPO 解决了 DPO 长度偏差并显著提升泛化 | Week 16 failure-mode sweep | 评价量与 IPO mean-logp 目标同构；历史 split 有 50 个 prompt 组泄漏；单 seed | UNVERIFIED | IPO 在当前目标同构指标上出现待重新切分确认的方向性信号 |
| P1-C07 | GRPO 获得真实迁移且不存在 reward hacking | Week 17 CMExam +2.2pp、unparseable=0；clean delta 仅可界定为 +0.61pp 至 +3.86pp | 500 题单 seed；8 题与 GRPO train 重叠；仅保留前 50 条预测，完整 clean CI/McNemar 不可恢复；仅排除部分格式攻击 | INVALIDATED | 训练 reward 上升；若 aggregate 可信，clean 点估计方向仍为正，但精确效应与确认性迁移未验证 |
| P1-C08 | response distillation 保知识优于真实答案 SFT | Week 19 三臂点估计 | 单 seed；解释长度、风格、正确性与信息密度同时变化 | SUPPORTED_TREND | 当前三臂结果提示 teacher explanation 可能减少附带退化 |
| P1-C09 | soft label 保知识、hard label 毁知识 | Week 19/20 跨实验比较 | hard 条件并非严格同数据/同目标对照；单 seed、多重比较 | UNVERIFIED | 当前 soft-KL 臂在所测开发集上更稳定，机制待受控验证 |
| P1-C10 | 纯 KL 优于含 hard CE，alpha=0 最佳 | Week 20 六臂点估计 | 单 seed；在同一 dev 上多臂选优 | UNVERIFIED | `kd_pure` 是当前开发集上的点估计 winner |
| P1-C11 | teacher judge 导致 reward hacking | Week 20 `rs_teacher` CMExam −0.2pp | 仅一个 seed，差异为 1/500；judge 目标与任务目标不一致 | UNVERIFIED | teacher-only 选择在该次运行未改善 CMExam，存在 proxy mismatch 假设 |
| P1-C12 | 50% 合成数据对真实数据非劣效 | Week 21 delta −0.8pp，CI [−3.4,+1.8]pp | CI 下界低于 −2pp；历史训练集存在 12 条高相似真实记录；clean matched v1 已建但尚未重训；非临床 AI 抽检 | INVALIDATED | 历史点估计有希望，但统计非劣效未建立；clean v1 只代表确认数据就绪 |
| P1-C13 | 合成数据质量已完成临床/人工验证 | Week 21 Codex 复核 30 条 | 不是人类或临床审阅 | RETRACTED | 完成了 30 条 AI 非临床 sanity check |

## 使用规则

- `VERIFIED` 才能无保留进入阶段总结。
- `SUPPORTED_TREND` 必须带“在当前配置/开发集/单次运行中”等范围限定。
- `UNVERIFIED` 不得改写成事实；`RETRACTED` 只能作为错误历史出现。
- 新确认实验使用新 claim ID；不得直接擦除旧 claim 的问题记录。
