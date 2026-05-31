# Phase 0 知识图谱

```
LLM Domain Adaptation
├── 基础
│   ├── PyTorch
│   │   ├── autograd (计算图 / backward / grad_fn)
│   │   ├── 训练循环 (forward → loss → backward → step)
│   │   └── 混合精度 (BF16 / FP16 / AMP)
│   ├── Transformer
│   │   ├── Attention (QKV / causal mask / scaled dot-product)
│   │   ├── FFN (Linear → GELU → Linear)
│   │   ├── Residual (Pre-LN vs Post-LN)
│   │   └── LayerNorm (手写实现)
│   └── 训练概念
│       ├── LR schedule (warmup + cosine decay)
│       ├── Gradient clipping
│       └── Perplexity = exp(loss)
├── 微调
│   ├── 全量微调
│   │   ├── 显存需求 (参数 × 6-8 bytes)
│   │   └── 何时使用 (数据量足够、算力充足)
│   ├── LoRA
│   │   ├── SVD 视角 (低秩分解)
│   │   ├── rank 选择 (8/16/32)
│   │   ├── alpha / scaling
│   │   └── target_modules (q/v vs 全部)
│   ├── QLoRA
│   │   ├── NF4 (4-bit Normal Float)
│   │   ├── Double Quantization
│   │   └── Paged Optimizer
│   └── SFT
│       ├── Chat Template (ChatML / Llama-3 / Mistral)
│       ├── Loss Masking (ignore_index = -100)
│       └── 数据质量 > 数量 (LIMA)
├── 评估
│   ├── Benchmark
│   │   ├── lm-eval-harness (MMLU / CMB)
│   │   └── 基线记录 (训练前后对比)
│   ├── LLM-as-Judge
│   │   ├── Pairwise comparison
│   │   └── Bias 分析 (位置 / 长度)
│   └── 人工评估
│       ├── Rubric 设计 (准确性 / 完整性 / 安全性 / 可读性)
│       └── IAA (Cohen's Kappa)
└── 数学
    ├── Attention 反向传播
    ├── Softmax + CE 梯度 (p_i - 1(i=y))
    ├── LoRA SVD 视角
    ├── DPO Loss 推导 (从 RLHF 到闭式目标)
    └── AdamW 更新规则 (解耦 weight decay)
```

## Gap 清单

### Gap 1: DPO/RLHF 缺乏实战经验
- **现状**: 推导了 DPO loss 的完整数学（Week 7），理解 Bradley-Terry 模型和配分函数消除
- **Gap**: 没有实际跑过 DPO 训练。4bit-QLoRA 项目的 `config/dpo.py` 和 `scripts/train_dpo.py` 已搭好框架但未执行
- **Phase 1 计划**: 用 master_data 的匹配结果构造偏好对（正确匹配 vs 错误匹配），跑 DPO 训练

### Gap 2: 分布式训练和大规模部署
- **现状**: 所有训练都在单机完成（Mac 64GB MLX / 单卡 GPU）
- **Gap**: 没有 DeepSpeed / FSDP / 多卡训练经验；没有 vLLM / TGI 等生产部署经验
- **Phase 1 计划**: 学习 FSDP 基础，尝试在多 GPU 环境训练更大模型

### Gap 3: 开放域评估能力不足
- **现状**: 评估体系以结构化任务为主（Top-1 accuracy、exact match）
- **Gap**: 没有实践过 LLM-as-Judge 的完整流程（只有代码模板）；没有做过人工评估 IAA
- **Phase 1 计划**: 在开放域任务（如医疗问答）上实践 LLM-as-Judge + 人工评估

### Gap 4: 数据工程深度不够
- **现状**: 用 Python 脚本做数据清洗和负采样，流程可用
- **Gap**: 没有系统化的数据质量监控（如自动化的数据分布分析、污染检测、难度分级）
- **Phase 1 计划**: 建立数据质量 dashboard，自动化难度分级和去重检测
