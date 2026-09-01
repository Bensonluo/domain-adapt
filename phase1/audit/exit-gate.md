# Phase 1 退出门槛

## 两条完成线

- **学习完成**：代码、笔记或实验流程跑通，并能解释结果。这一层大部分已完成。
- **研究通过**：结论由独立、可复现、能回答该问题的证据支持。这一层尚未通过。

周级状态必须分别记录：`DELIVERABLE_DONE / EVIDENCE_PRESENT / REPRODUCIBLE / CLAIM_VERIFIED / FOLLOWUP_REQUIRED`。勾选交付物不能自动推出 claim 已验证。

## 通过条件

Phase 1 只有同时满足以下条件，才可从 `REMEDIATION_REQUIRED` 改为 `PASSED`：

1. `phase1/README.md`、claim matrix、artifact manifest 和最终总结状态一致。
2. 核心数据、base model revision、训练配置、seed、checkpoint、逐题预测和 summary 有稳定 lineage。
3. 当前 CMExam 500 题及既有 CMMLU 子集明确标为 dev；另有冻结且未参与选模的 blind test。
4. 核心 winner 至少 3 个独立训练 seed；报告 paired bootstrap CI 和适用时的 McNemar，而非只比较点估计。
5. CPT 的方法、模型、数据、token budget 和训练步数按待回答问题受控；不得用同时更换多个变量的运行归因给 LoRA 或数据配比。
6. 偏好数据按规范化 prompt/source 分组切分；长度、来源和重复泄漏通过审计；DPO/IPO 包含匹配的 SFT/instruct baseline。
7. GRPO 结论包含 reward-hacking probes、独立 outcome metric 和至少一个关键超参消融；只能声称已排除实际检查过的 hacking 类型。
8. hard CE vs soft KL 使用同数据、同 token budget、同训练制度和多 seed 对照；“dark knowledge”等机制解释需独立验证，不能由多臂 winner 反推。
9. 合成数据非劣效 margin 在看结果前冻结；移除或分层报告 overlap；临床正确性声明由人类临床评审支持。
10. 阶段最终总结中的每个结果性声明映射到 claim ID；核心 claim 为 `VERIFIED` 或正式 `WAIVED`。

## 当前判定

`REMEDIATION_REQUIRED`：Week 9–21 学习与实验执行丰富，但独立盲测、多 seed、受控归因和跨周证据链仍未闭环。
