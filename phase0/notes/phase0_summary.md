# Phase 0 总结：LLM Domain Adaptation 基础

> 2026-05 完成，历时 8 周

---

## 学习路径回顾

### Week 1-2: PyTorch + Transformer 基础

**核心收获**: 从零理解 autograd 和 attention 机制

- 手写 `backward_manual` 并与 `autograd` 结果对比，理解计算图和梯度传播
- 手写 Multi-Head Attention、LayerNorm、完整 Transformer block
- 理解 Pre-LN vs Post-LN、因果掩码、位置编码
- 关键文件: `phase0/week1/day5_7_transformer.py`

**能力建立**: 能在白板上画出 Transformer 的完整数据流和 shape 变化

---

### Week 3: nanoGPT 训练

**核心收获**: 理解完整的 LLM 训练循环

- 从 Andrej Karpathy 的 nanoGPT 出发，理解 tokenization、训练循环、LR schedule
- 手动调超参（learning rate、batch size、gradient clipping）观察 loss 曲线变化
- 理解 perplexity = exp(loss) 的含义

**能力建立**: 能独立搭建一个可运行的训练循环

---

### Week 4: LoRA 深度理解

**核心收获**: 从数学（SVD）到工程（PEFT 源码）完整掌握 LoRA

- 手写 LoRA from scratch：`W = W_0 + (α/r) × B × A`
- SVD 视角：分析 Qwen2.5-3B 权重的奇异值衰减，验证低秩假设
- QLoRA 三大技术：NF4 量化 + Double Quantization + Paged Optimizer
- 关键文件: `phase0/week4/lora_from_scratch.py`

**能力建立**: 能解释 LoRA 为什么有效（ΔW 的秩远低于 W_0），能选择 rank/alpha/target_modules

---

### Week 5: SFT 细节

**核心收获**: 理解 SFT 的每个组件 — template、masking、数据质量

- Chat Template 对比：Qwen/ChatML、Llama-3、Mistral 三种格式的差异
- Loss Masking 手写实现：`mask_labels` 函数，只在 assistant 回复上计算 loss
- 完整 SFT 训练脚本：整合 template + masking + QLoRA
- SFT Checklist：LR 2e-4（QLoRA）、1-3 epochs、rank 8-16
- 关键文件: `phase0/week5/loss_masking.py`

**能力建立**: 能诊断 SFT 训练中的常见 bug（如 loss 正常但模型重复 prompt）

---

### Week 6: 完整领域模型训练

**核心收获**: 端到端训练一个可用的领域模型

- 实战项目: [4bit-QLoRA-post-training](https://github.com/luopeng/4bit-QLoRA-post-training)
- 数据：从 14K+ 药品知识库生成 58K+ 训练样本，含硬负采样和噪声注入
- 训练：7 个预设（Mac 64GB MLX + GPU），覆盖 1.7B 到 14B 模型
- 评估：分难度 accuracy 对比，10+ 轮迭代
- 关键文件: `4bit-QLoRA/domains/master_data/scripts/train.py`

**能力建立**: 能独立完成"数据准备 → 训练 → 评估 → 迭代"的完整闭环

---

### Week 7: 数学推导密集周

**核心收获**: 建立 LLM 训练核心数学的直觉

5 个推导全部完成：

1. **Self-Attention 反向传播**: ∂L/∂V, ∂L/∂S, ∂L/∂Q, ∂L/∂K；理解 √d_k 缩放的数学依据
2. **Softmax + CE 梯度**: ∂L/∂z = p - y_onehot（概率减 one-hot = 预测误差）
3. **LoRA SVD 视角**: Eckart-Young 定理、能量捕获比、alpha/rank 解耦设计
4. **DPO Loss**: 从 RLHF 目标 → 闭式解 → 反解 reward → Bradley-Terry → 消掉 Z(x)
5. **AdamW 更新规则**: 解耦 weight decay，decay 不经过 m/v 直接作用于参数

**能力建立**: 推导不用记住，但需要理解"结果为什么长这样"

---

### Week 8: 评估方法论

**核心收获**: 建立三层评估体系

1. **Benchmark 评估**: lm-evaluation-harness，检测灾难性遗忘
2. **结构化评估**: 自动化 Top-1 accuracy + F1 + grade accuracy
3. **跨模型对比**: 9 个模型的排行榜，验证 SFT 的 ROI

关键结论：26B finetuned 模型在领域任务上超越 35B baseline 和商业云端 API

---

## 实战项目成果

### [4bit-QLoRA-post-training](https://github.com/luopeng/4bit-QLoRA-post-training)

| 维度 | 成果 |
|------|------|
| **数据** | 14K 药品知识库 → 58K+ 训练样本（硬负采样 + 噪声注入 + 零泄漏划分） |
| **训练** | 7 个模型预设，Mac MLX + GPU 双路径，MLflow + TensorBoard 追踪 |
| **评估** | 74 次评估记录，9 个模型排行榜，分难度分析 |
| **核心结果** | gemma-4-26b Institution Top-1: 79.75% → **98.75%** (+19%) |
| **工程** | 完整的 CLI 工具链 + Streamlit dashboard + LoRA merge 脚本 |

---

## 已建立的能力

- ✅ Transformer 架构理解（能读源码、能调 shape）
- ✅ LoRA/QLoRA 实战（能手写、能调参、能解释为什么有效）
- ✅ SFT 全流程（数据 → template → masking → 训练 → 评估）
- ✅ 评估方法论（benchmark + 结构化评估 + 跨模型对比）
- ✅ Mac 本地 MLX 训练 + GPU QLoRA 训练
- ✅ 数学直觉（attention 梯度、DPO、AdamW 的"为什么"）

## 仍需加强的领域

详见 [knowledge_graph.md](../week8/knowledge_graph.md) 的 Gap 清单：
1. DPO/RLHF 实战（推导懂了，没跑过）
2. 分布式训练（DeepSpeed/FSDP）
3. 开放域评估（LLM-as-Judge + 人工 IAA）
4. 数据工程自动化（质量监控、难度分级）

---

## Phase 1 切入点

### 方向 A: DPO 偏好优化（推荐优先）
- 用匹配结果构造偏好对（正确 vs 错误），跑 DPO 训练
- 已有代码框架: `4bit-QLoRA/config/dpo.py`
- 预期: 提升置信度校准，减少"高置信度但错误"的 case

### 方向 B: 多领域扩展
- 从主数据匹配扩展到药品相互作用、处方审核等场景
- 验证 SFT 方法论的可迁移性

### 方向 C: 生产部署
- MLX → vLLM / TGI 部署
- 量化生产环境的延迟和吞吐量需求
- 建立持续评估 pipeline
