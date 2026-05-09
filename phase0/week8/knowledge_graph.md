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

## 能力自检

- [ ] 能手写 backward_manual 并和 autograd 一致
- [ ] 能白板画出 attention 的完整 shape 流
- [ ] 能解释 Pre-LN 为什么比 Post-LN 稳定
- [ ] 能推导 LoRA 的 SVD 视角
- [ ] 能解释 QLoRA 的三个关键技术
- [ ] 能手写 mask_labels 函数
- [ ] 能推导 DPO loss (从 RLHF 到闭式)
- [ ] 能解释 AdamW 和 Adam 的区别
- [ ] 能设计完整的人工评估 rubric
- [ ] 能用 lm-eval 跑 benchmark 并解读结果

## Gap 清单

TODO: 完成 Phase 0 后填写
- [ ] Gap 1: ...
- [ ] Gap 2: ...
- [ ] Gap 3: ...
