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

#### 逐模块对比

| 方面 | Week 1 MiniGPT | Week 2 nanoGPT | 差异原因 |
|------|---------------|----------------|---------|
| **Tokenizer** | char-level (65 vocab) | BPE tiktoken (50304 vocab) | BPE 信息密度高，同样文本 token 数少 3-4x，训练更快 |
| **vocab_size** | 65 (字符集大小) | 50304 (pad 到 64 倍数) | GPU tensor core 要求维度对齐，50304 而非 50257 |
| **Weight init** | N(0, 0.02) 全部统一 | N(0, 0.02) + 残差投影缩放 `1/√(2·n_layer)` | 深层网络残差通路方差会累积，缩放保证前向/反向方差稳定 |
| **LR schedule** | 固定 3e-4 + 线性 warmup | warmup + cosine decay 到 min_lr | cosine decay 让模型后期收敛更稳定 |
| **Attention** | 手写 QK^T + mask + softmax | `F.scaled_dot_product_attention` (Flash Attention 可选) | Flash Attention 不显式存 attention 矩阵，省显存 O(N) → O(√N) |
| **QKV 投影** | 3 个独立 `nn.Linear` | 1 个合并 `nn.Linear(D, 3D)` + `chunk` | 合并后单次矩阵乘，GPU 利用率更高 |
| **数据加载** | `torch.stack` 全量加载 | `np.memmap` 内存映射 | memmap 不用把整个数据集加载到内存，适合大语料 |
| **混合精度** | 无 (FP32) | 支持 `torch.amp` BF16/FP16 | 显存减半 + 速度翻倍，BF16 不需要 loss scaler |
| **梯度裁剪** | `clip_grad_norm_(max_norm=1.0)` | 同上 | 两边一样，防止某些 batch 梯度爆炸 |
| **Weight tying** | `head.weight = tok_emb.weight` | 同上 | 共享参数省 ~25% 显存，两边的做法一致 |
| **数据传输** | `.to(device)` | `.pin_memory().to(device, non_blocking=True)` | pin_memory + non_blocking 让 CPU→GPU 传输和计算重叠 |
| **Checkpoint** | 保存 state_dict | 保存 best + final，按 val loss 选 best | Week 1 只保存一个，nanoGPT 保留最优 |
| **生成采样** | temperature + multinomial | temperature + top-k + multinomial | top-k 过滤低概率 token，减少乱码 |

#### 关键认知

1. **`vocab_size=50304`** — GPT-2 BPE 词表实际 50257 个 token，但 50304 是最近的 64 倍数。GPU tensor core 按 64×64 分块矩阵乘，对齐后计算效率 ~30% 提升，多出来的 token 不用就是浪费

2. **残差投影缩放** — `c_proj`（每个 Block 的 attention 输出投影和 FFN 输出投影）的 weight 初始化时除以 `1/√(2·n_layer)`。原因：每个 Block 有 2 个残差加法（attn + ffn），N 层就有 2N 次叠加，缩放让输出方差保持 ~1

3. **BPE vs char-level 的实际影响**:
   - "患者主诉头痛" char-level = 6 tokens，BPE 可能 = 2-3 tokens
   - 同样的 `block_size=256`，BPE 能看到 3-4x 更长的语义跨度
   - 代价：vocab 大 770x，embedding 层参数多

4. **memmap vs 全量加载** — Week 1 的 tiny_shakespeare 只有 1MB 直接全加载没问题。但真实语料几百 MB 到几 GB，memmap 让操作系统按需加载，内存占用 = batch 大小而非语料大小


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
