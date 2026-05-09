# Week 2: nanoGPT 逐行理解 + 领域数据训练

> 目标: 理解 production-quality 的小型 GPT 实现每一个细节,在领域数据上跑通训练。
> 预计时间: 14-20 小时

> **上周回顾**: Week 1 你手写了 Transformer 的核心模块 — autograd、attention、FFN、MiniGPT。但那是一个最简化的教学实现。这周你要看 Karpathy 的 nanoGPT,理解"从玩具到可复现研究"的工程差距在哪里。
>
> **为什么学这周**: nanoGPT 是你从"理解原理"到"能跑真实实验"的桥梁。它覆盖了完整训练循环的每个工程细节 — 数据加载、LR 调度、混合精度、checkpoint。理解它之后,你在 Week 3 看 HuggingFace Trainer 就能分清"哪些是工程必要"和"哪些是框架抽象"。
>
> **思考锚点**: "从 MiniGPT 到 nanoGPT,哪些改动是性能优化,哪些是正确性保障?"

---

## Day 1-3: nanoGPT 逐行阅读

> **思考**: nanoGPT 的 vocab_size=50304 而不是 GPT-2 的 50257。为什么要 pad 到 64 的倍数? (提示: GPU tensor core 的计算效率)

### 做什么
1. `git clone https://github.com/karpathy/nanoGPT`
2. 逐行阅读 `model.py` (~200 行):
   - `GPTConfig` 数据类
   - `CausalSelfAttention` 类
   - `Block` 类 (Transformer Block)
   - `GPT` 类: `__init__`, `forward`, `estimate_mfu`
3. 逐行阅读 `train.py` (~300 行):
   - 数据加载 (`np.memmap`)
   - 学习率调度 (warmup + cosine decay)
   - 梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)
   - 混合精度 (`torch.amp`)
   - DDP (分布式数据并行,了解概念即可)
4. 阅读 `sample.py`: temperature、top-k 采样

### 对比 Week 1 — 找差异
| 方面 | 你的 MiniGPT | nanoGPT |
|------|-------------|---------|
| Tokenizer | char-level | BPE (tiktoken) |
| Weight init | N(0, 0.02) 全部 | 残差投影缩放 1/√N |
| LR schedule | 固定 | warmup + cosine decay |
| Attention | F.scaled_dot_product_attention | Flash Attention (可选) |

这个对比表是你的核心笔记,能画出来说明你理解了。

### 交付物
- `phase0/notes/week2_nanogpt_reading.md` — 逐行注释笔记
- `phase0/notes/week2_key_concepts.md` — 5 个关键概念整理

---

## Day 4-5: 在领域数据上训练

> **思考**: 为什么 nanoGPT 用 BPE tokenization 而不是 char-level? 对训练效率和生成质量各有什么影响?

### 做什么
1. 收集 10-50MB 工作领域文本 (临床指南/行业文档/业务记录)
2. 准备数据: `python data_prep.py`
3. 配置训练参数 (1-10M 参数模型):
   ```python
   n_layer=4, n_head=4, n_embd=128, block_size=256, batch_size=64, lr=3e-4, max_iters=5000
   ```
4. 启动训练,观察 loss 曲线
5. 尝试生成: `python sample.py`, 用不同 temperature (0.5 / 0.8 / 1.0)

### 跑
```bash
# 准备数据
python phase0/week2/data_prep.py --input /path/to/domain_text.txt --output phase0/data/processed/domain.bin

# 训练 (Mac MPS/CPU,小模型)
python phase0/week2/train_nanogpt.py --data phase0/data/processed/domain.bin --out_dir phase0/checkpoints/nanogpt_domain

# 生成样例
python phase0/week2/sample.py --checkpoint phase0/checkpoints/nanogpt_domain/ckpt.pt --temperature 0.8
```

### 交付物
- `phase0/results/week2_loss_curve.png` — loss 曲线
- `phase0/results/week2_samples.txt` — 生成文本样例 (好/坏各 3 条 + 分析)

---

## Day 6-7: 关键概念整理

> **思考**: val loss 开始上升意味着什么? 这时候继续训练会怎样?

### 做什么
1. 理解 "next token prediction" 为什么能学到语言 (交叉熵损失的含义)
2. 理解 perplexity = `exp(average_loss)`, 越低越好
3. 理解 train/val split: val loss 开始上升 = 过拟合信号
4. 写笔记: 训练循环每个组件的作用 (1 页 A4 纸)

### 交付物
- `phase0/notes/week2_training_components.md` — 训练循环组件清单
- `phase0/notes/week2_perplexity.md` — perplexity 与过拟合理解

---

## 自测题

1. **nanoGPT 的学习率调度分几个阶段?** warmup 阶段 LR 怎么变? cosine decay 阶段呢?
2. **混合精度训练用 BF16 和 FP16 的关键区别是什么?** 为什么 BF16 不需要 loss scaler?
3. **`gradient clipping` 的 `max_norm=1.0` 是什么意思?** 为什么要做梯度裁剪?

> 答案: 1) warmup 阶段从 0 线性增到 max_lr;之后 cosine decay 到 min_lr;最后保持 min_lr。2) BF16 的指数位和 FP32 一样宽(8 bit),不会溢出,所以不需要 loss scaler;FP16 指数位只有 5 bit,容易上溢/下溢。3) 把梯度向量的 L2 范数裁剪到不超过 1.0;防止某些 batch 的梯度爆炸导致训练不稳定。

---

## 验收清单

- [ ] nanoGPT `model.py` 逐行注释完成
- [ ] nanoGPT `train.py` 逐行注释完成
- [ ] 领域数据训练跑通,loss 下降
- [ ] 生成样例存档 (不同 temperature)
- [ ] 训练循环组件笔记 (1 页 A4)
- [ ] 自测题能回答 2/3 以上
