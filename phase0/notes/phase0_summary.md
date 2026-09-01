# Phase 0 总结：LLM Domain Adaptation 基础

> 2026-05 完成 8 周学习执行；2026-08-28 方法论复审
>
> 当前状态：`REMEDIATION_REQUIRED`。学习性主体完成，研究性验收未通过。

阶段状态以 [Phase 0 状态页](../README.md) 为唯一来源；错误形成原因保存在每周 `CORRECTIONS.md`。

## 准确结论

Phase 0 已建立 Transformer、训练循环、LoRA/QLoRA、SFT 数据目标和结构化评估的基础理解，并完成一个外部结构化匹配 POC。

当前最强但仍受范围限制的结果是：

> 在同源合成的机构匹配评估上，“Gemma 26B 微调模型 + 当时 MLX 推理链路”相对“基座模型 + 当时 LM Studio 链路”观察到 Top-1 79.75%→98.75%。

该观测值不等于 LoRA/SFT 的纯因果贡献，也不能直接外推到真实医疗业务、统一条件的跨模型排名或推理延迟收益。对应声明边界见 [claim-evidence matrix](../audit/claim-evidence-matrix.md)。

## 学习路径与证据状态

### Week 1：PyTorch + Transformer 基础

- 手写 autograd、attention、Transformer block 和 toy 训练。
- 代码、loss 图、sample 和 checkpoint 存在。
- 梯度/一致性测试输出、lm-eval 基线和脱稿能力证据未统一留存。

状态：`PARTIAL`。

### Week 2：nanoGPT 训练

- 完成训练脚本、不同 temperature 样例和训练 takeaways。
- 原总结曾误写为 Week 3，现已纠正。

状态：`MOSTLY_DONE / EVIDENCE_PARTIAL`。

### Week 3：HuggingFace 源码 + 全量微调

- HF/nanoGPT、Trainer 对比材料和 full-FT 脚本存在。
- 没有 full-FT 运行日志、峰值显存、硬件/时间和 loss 曲线，不能声称已经亲身完成全量微调实验。

状态：`PARTIAL`。

### Week 4：LoRA / QLoRA

- 手写 LoRA、PEFT 对比和论文笔记存在。
- `lora.ipynb` 是随机矩阵 SVD 教学演示，不是 Qwen2.5-3B 或真实训练 ΔW 分析。
- LoRA 的经验动机是任务更新具有较低内在维度，不要求预训练权重 W_0 本身低秩。
- rank 8/16/32 的适用性尚未由本项目受控实验验证。

状态：`IMPLEMENTATION_DONE / SVD_CLAIM_INVALIDATED`。

### Week 5：Chat Template + Loss Masking

- 完成 template tokenization 对比、masking 标签实现和 SFT 脚本。
- 已修正训练目标：assistant 内容及 turn-ending token 参与 loss；marker 缺失时 fail-closed。
- 尚未完成 masking vs unmasked 的模型效果对照。
- 原“质量 > 数量”设计同时改变质量、数量和 token budget，且无结果报告，结论已撤回。

状态：`PARTIAL`。

### Week 6：领域 SFT 替代案例

外部 `4bit-QLoRA-post-training/medical_entity` 提供了药品实体匹配数据、训练和结构化评估案例。这是有价值的替代性交付，但没有等价完成原计划的开放式领域 SFT 与人工评估。

状态：`EXTERNAL_SUBSTITUTE / PARTIAL`。

### Week 7：数学推导

attention、softmax+CE、LoRA/SVD、DPO 和 AdamW 五份推导文档存在。文档完成与白板脱稿复述能力是两种证据；后者尚未正式验收。LoRA/SVD 中不存在的 Qwen ΔW 数字和过强 rank 结论已撤回。

状态：`DERIVATIONS_PRESENT / VALIDITY_PARTIAL`。

### Week 8：评估方法论

- 完成独立 `master_data/Gemma 26B` 案例的结构化 ground-truth 评估。
- LLM-as-Judge 只有原型代码；没有运行结果或 judge 校准。
- 人工 rubric 是模板；没有盲化评分和 IAA。
- `master_data` 不是 Week 6 `medical_entity` 模型的后续评估，两个案例不能合并成同一闭环。

状态：`STRUCTURED_EVAL_PRESENT / FULL_EVAL_INCOMPLETE`。

## 两个外部案例必须分开

| 案例 | 模型/任务 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `medical_entity` | Qwen 系列、药品实体匹配 | 完成过真实的数据→SFT→结构化评估工程流程 | 原计划开放式医疗问答能力和人工评估 |
| `master_data` | Gemma 26B、机构/产品匹配 | 在当时同源合成评估和两条完整推理链路间观察到强提升 | LoRA 单独因果贡献、真实业务外部效度、统一跨模型排名和延迟收益 |

## 已建立的能力

- Transformer 与训练循环：有代码证据，行为验收部分缺失。
- LoRA/QLoRA：实现和论文理解已建立；rank/SVD 经验结论待验证。
- SFT：template、masking 和训练脚本已建立；关键消融待补。
- 结构化领域 POC：已建立，但主要是同源合成分布。
- 评估：结构化自动评估已实践；开放式 judge、人类评估和 IAA 尚未闭环。

## 进入正式验收前的最小工作

1. 同 runtime、量化、prompt、解码条件重跑 base vs adapter，并使用新 blind test。
2. 完成 masking 三组消融和质量/数量解耦实验，至少 3 个 seed。
3. 用 full-FT 的真实 ΔW 做多层 SVD，并与 rank 消融连接。
4. 保存逐题预测，报告 paired CI/McNemar；旧的反复使用测试集降级为 dev。
5. 若仍声称覆盖开放式领域能力，补盲化人工双评审与 IAA；LLM judge 仅作补充。

详细协议见 [最小重跑计划](../audit/rerun-plan.md) 和 [退出门槛](../audit/exit-gate.md)。
