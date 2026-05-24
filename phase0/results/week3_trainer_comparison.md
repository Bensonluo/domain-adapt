# Week 3: nanoGPT 训练循环 vs HF Trainer 对比

## 1. 训练主循环

```python
# nanoGPT — 手写 ~60 行
for step in range(max_iters):
    lr = get_lr(step, warmup_iters, lr_decay_iters, max_lr, min_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    x, y = get_batch(data, "train", config, batch_size, block_size, device)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

```python
# HF Trainer — 一行
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

结论：Trainer 内部做的事和 nanoGPT 一模一样。区别是 nanoGPT 是代码，HF 是配置。

## 2. 学习率调度

```python
# nanoGPT — 手写函数
def get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

```python
# HF — 配置项
training_args = TrainingArguments(
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    learning_rate=5e-5,
)
```

结论：nanoGPT 自己算每步的 lr，HF 用 `lr_scheduler_type` 选策略。HF 内部调的也是同样的 cosine 公式。

## 3. 梯度累积

```python
# nanoGPT — 手动拆分
for micro_step in range(grad_accum_steps):
    x, y = get_batch(...)
    with ctx:
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
    loss.backward()
    if (micro_step + 1) % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

```python
# HF — 配置项
training_args = TrainingArguments(
    gradient_accumulation_steps=4,
)
# Trainer 内部自动处理，逻辑完全一样
```

结论：同样的 micro batch 拆分 + loss 归一化 + 累积够了才 step。

## 4. 混合精度

```python
# nanoGPT — 手动上下文
ctx = torch.amp.autocast(device_type, dtype=torch.bfloat16)
with ctx:
    logits, loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

```python
# HF — 配置项
training_args = TrainingArguments(
    bf16=True,  # 或 fp16=True
)
# Trainer 内部自动用 autocast + GradScaler（fp16 时）
```

结论：nanoGPT 要自己管 scaler（fp16 时），HF 一行配置搞定。

## 5. 评估

```python
# nanoGPT — 手写 eval 函数
@torch.no_grad()
def estimate_loss(model, train_data, val_data, ...):
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, ...)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

```python
# HF — 配置项
training_args = TrainingArguments(
    eval_strategy="epoch",    # 或 "steps"
    eval_steps=200,
)
# Trainer 自动在训练中穿插 eval，计算 eval_loss
```

结论：nanoGPT 每 N 步手动调 eval 函数，HF 按 strategy 自动触发。

## 6. 保存 Checkpoint

```python
# nanoGPT — 手动判断最优
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best.pt")
torch.save(model.state_dict(), "final.pt")
```

```python
# HF — 配置项
training_args = TrainingArguments(
    save_strategy="epoch",
    save_total_limit=2,          # 只保留最近 2 个 checkpoint
    load_best_model_at_end=True, # 训练结束加载最优权重
)
# Trainer 自动保存，自动清理旧的，自动选 best
```

结论：nanoGPT 只存 best 和 final。HF 多了 `save_total_limit`（自动清理磁盘）和 `load_best_model_at_end`。

## 7. 数据加载

```python
# nanoGPT — np.memmap + 手动 batch
train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
ix = torch.randint(len(train_data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy(train_data[i:i+block_size]) for i in ix])
```

```python
# HF — Dataset + DataCollator
dataset = Dataset.from_json("data.jsonl")
dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# Trainer 内部自动: shuffle → batch → pad → 移到 GPU
```

结论：nanoGPT 假设数据等长（固定 block_size），不需要 pad。HF 要处理变长序列，所以需要 DataCollator 自动 pad + 生成 labels。

## 总结

| 手写的代码（Week 1-2） | HF Trainer 配置项 |
|-------------------------|-------------------|
| `for step in range(N)` | `num_train_epochs=3` |
| `get_lr(step, ...)` | `lr_scheduler_type="cosine"` |
| `loss / grad_accum_steps` | `gradient_accumulation_steps=4` |
| `torch.amp.autocast(...)` | `bf16=True` |
| `clip_grad_norm_(...)` | `max_grad_norm=1.0` |
| `estimate_loss(...)` | `eval_strategy="epoch"` |
| `torch.save(...)` | `save_strategy="epoch"` |
| `np.memmap + get_batch` | `Dataset + DataCollator` |

**本质**：每一行手写代码都能在 TrainingArguments 里找到一个对应配置。Trainer 没有魔法，就是把训练循环参数化了。
