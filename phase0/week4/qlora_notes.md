# QLoRA 论文精读笔记

> 论文: QLoRA: Efficient Finetuning of Quantized LLMs (2023)
> 链接: https://arxiv.org/abs/2305.14314

---

## 核心问题

为什么 4-bit 量化 + LoRA 能和 16-bit 全量微调几乎没差异?

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

### Figure 1: QLoRA vs 16-bit FT

TODO: 记录你看到的结论
- [ ] 哪些 benchmark 上 QLoRA 和 16-bit FT 差距 < 1%?
- [ ] 哪些 benchmark 差距稍大? 为什么?

---

## 显存对比 (手写)

| 方案 | 模型权重 | 梯度 | 优化器状态 | 总计 |
|------|----------|------|------------|------|
| 16-bit 全量微调 | 2B × 2 | 2B × 2 | 2B × 4 | ? |
| LoRA (16-bit) | 2B × 2 | ? | ? | ? |
| QLoRA | ? | ? | ? | ? |

TODO: 填完上表,理解 QLoRA 为什么能在单卡 48GB 上微调 65B 模型。

---

## 问题清单

- [ ] NF4 的量化/反量化过程是 deterministic 的吗?
- [ ] Double Quant 的第二次量化用几 bit?
- [ ] Paged Optimizer 对训练速度的影响?
