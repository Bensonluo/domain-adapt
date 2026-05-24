# Week 2: nanoGPT 核心提炼

> 模型架构和 Week 1 MiniGPT 一样，真正的价值在训练工程技巧。

## 从 MiniGPT 到 nanoGPT 的 5 个关键升级

### 1. 残差投影缩放

```python
# nanoGPT model.py
nn.init.normal_(module.weight, mean=0.0, std=0.02)
if isinstance(module, nn.Linear) and module.weight.shape == (n_embd, n_embd):
    module.weight.data *= 1.0 / math.sqrt(2 * config.n_layer)
```

Week 1 MiniGPT 没做这个。每个 Block 有 2 个残差加法（attn + ffn），N 层叠加 2N 次，缩放让输出方差保持 ~1。深层网络必须做，否则残差通路方差爆炸。

### 2. Cosine LR Decay（三阶段调度）

```python
# warmup: 0 → max_lr（线性增长）
if it < warmup_iters:
    return max_lr * (it + 1) / (warmup_iters + 1)
# cosine decay: max_lr → min_lr
decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
return min_lr + coeff * (max_lr - min_lr)
# floor: 保持 min_lr
if it > lr_decay_iters:
    return min_lr
```

Week 1 train_toy.py 只有 warmup 没有衰减。Cosine decay 让模型后期收敛更稳定。

### 3. 混合精度训练

```python
ctx = torch.amp.autocast(device_type, dtype=torch.bfloat16)
with ctx:
    logits, loss = model(x, y)
```

一行代码，显存减半 + 速度翻倍。BF16 指数位和 FP32 一样宽（8 bit），不会溢出，不需要 loss scaler。FP16 指数位只有 5 bit，需要 GradScaler。

Week 1 训练没用混合精度。

### 4. 梯度累积（模拟大 batch）

```python
for micro_step in range(grad_accum_steps):
    x, y = get_batch(...)
    with ctx:
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # 除以累积步数
    loss.backward()
    if (micro_step + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

GPU 显存不够跑大 batch？拆成多个 micro batch 梯度累积，效果等价于大 batch。关键：loss 要除以累积步数。

### 5. Eval + Save Best

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), out_dir / "best.pt")
```

按 val loss 存最优模型，不是最后一步的模型。Week 1 只存了 final，可能过拟合了。

---

## 不需要深究的部分

| 内容 | 原因 |
|------|------|
| DDP 分布式训练 | 单卡 4090 用不上 |
| `estimate_mfu` | 调试用工具函数 |
| config 系统 | 纯工程，不影响理解 |
| `sample.py` top-k | Week 1 已学过 temperature + multinomial |

---

## 总结对照

| 你已经会的（Week 1） | nanoGPT 额外教的（Week 2） |
|---------------------|--------------------------|
| Transformer 架构 | 残差投影缩放 |
| 手写 attention | 混合精度训练 |
| 基础训练循环 | cosine LR + 梯度累积 |
| 保存 checkpoint | eval + save best |
| char-level tokenizer | BPE (tiktoken) + memmap |
