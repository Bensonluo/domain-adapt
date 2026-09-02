# QLoRA 论文精读笔记

> 论文: QLoRA: Efficient Finetuning of Quantized LLMs (2023)
> 链接: https://arxiv.org/abs/2305.14314

---

## 核心问题

4-bit 量化 + LoRA 在论文哪些模型、数据和评估设置下接近 16-bit 对照？哪些规模缺少同条件 full-FT 证据，不能外推为普遍等价？

---

## 三个关键技术

### 1. NF4 (4-bit Normal Float)

TODO: 用自己的话解释
- [ ] 为什么叫 "Normal Float"? 假设权重服从什么分布?
- [ ] 分位数量化是什么意思?
- [ ] 和 uniform 4-bit 量化比,为什么 NF4 更好?

### 2. Double Quantization

TODO: 用自己的话解释
- [ ] 第一次量化: 权重量化成 4-bit
- [ ] 第二次量化: 量化常数再量化一次
- [ ] 总共省了多少显存?

### 3. Paged Optimizer

TODO: 用自己的话解释
- [ ] 什么时候会触发 CPU 内存分页?
- [ ] 和梯度检查点 (gradient checkpointing) 的区别?

---

## 关键 Figure

### Figure 1：方法与内存可行性，不是普遍 full-FT 等价证明

TODO: 记录对比结论
- [ ] Figure 1 实际表达了哪些组件和内存主张？
- [ ] 性能数字分别来自哪些表、模型规模和 baseline？
- [ ] 33B/65B 上是否存在同条件 16-bit full FT 对照？若没有，明确写“未验证”。

---

## 显存对比 (手写)

| 方案 | 模型权重 | 梯度 | 优化器状态 | 总计 |
|------|----------|------|------------|------|
| 16-bit 全量微调 | 2B × 2 | 2B × 2 | 2B × 4 | ? |
| LoRA (16-bit) | 2B × 2 | ? | ? | ? |
| QLoRA | ? | ? | ? | ? |

TODO: 填表时同时记录序列长度、batch、激活、checkpointing、临时 buffer 和 kernel；不能把参数字节数之和当作实测峰值。用论文设置解释 65B/48GB 可行性，不外推固定显存比例。

---

## 问题清单

- [ ] NF4 的量化/反量化过程是 deterministic 的吗?
- [ ] Double Quant 的第二次量化用几 bit?
- [ ] Paged Optimizer 对训练速度的影响?
