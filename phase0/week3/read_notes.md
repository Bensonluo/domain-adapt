# Week 3 源码阅读笔记模板

## LlamaForCausalLM.forward

```python
# 源码位置: transformers/src/transformers/models/llama/modeling_llama.py
```

### LlamaForCausalLM.forward 问答

**Q: `input_ids` 怎么变成 embedding?**

```python
# LlamaModel.forward 内部
inputs_embeds = self.embed_tokens(input_ids)  # (B, T) → (B, T, D)
```
和 Week 1 的 `self.tok_emb(idx)` 完全一样，就是查 embedding 表。

**Q: `past_key_values` 是什么? KV cache 怎么工作的?**

```python
# 第一次 forward: past_key_values = None，算全部 K/V
# 返回时把每层的 K/V 存到 past_key_values 里
# 后续 forward: past_key_values 有值，只算新 token 的 K/V，和历史拼接
for layer in self.layers:
    hidden = layer(hidden, past_key_values=layer_past)
    # layer 内部: k = cat(past_k, new_k), v = cat(past_v, new_v)
```
生成时第 1 步算完整序列，第 2 步起只 forward 新 token。避免了每步重算 O(N²) 的 attention。

**Q: `lm_head` 和 `embed_tokens` 的 weight tying 在哪里?**

```python
# LlamaForCausalLM.__init__
self.model = LlamaModel(config)
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

# PreTrainedModel.tie_weights() 里
self.lm_head.weight = self.model.embed_tokens.weight
```
和 `self.head.weight = self.tok_emb.weight` 一模一样。HF 放在基类 `tie_weights()` 里统一处理。

**Q: `causal_mask` 在哪里生成的?**

```python
# LlamaModel.forward 内部
causal_mask = self._update_causal_mask(
    attention_mask, past_key_values, input_ids, ...
)
# 本质就是生成一个 (T, T) 的上三角 mask，和你 Week 1 的 triu 一样
# 但 HF 还要处理：padding mask（batch 内不等长）、KV cache 的 mask 长度
```
这就是为什么 HF 的 forward 长 — 它要处理 padding + cache + 多种输入组合。

**Q: `position_ids` 是自动生成的还是外部传入的?**

```python
# 如果外部没传，自动生成
if position_ids is None:
    position_ids = torch.arange(past_length, seq_length, device=device)
```
但有了 KV cache 后 `past_length > 0`，所以 position 从缓存长度开始而非 0。

---

## LlamaModel.forward

**Q: `embed_tokens` → 逐层 `LlamaDecoderLayer` → `norm` → `output` 的 shape 流?**

```
input_ids          (B, T)
embed_tokens       (B, T, D=2048)
layer 0 hidden     (B, T, 2048)
layer 1 hidden     (B, T, 2048)
...
layer N hidden     (B, T, 2048)
norm               (B, T, 2048)    ← 最终 hidden_states
```
和 MiniGPT 一模一样：embedding → N 个 Block → LayerNorm。每层 shape 不变。

---

## LlamaDecoderLayer

**Q: `self_attn` 的输入是什么?**

```python
# LlamaDecoderLayer.forward
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)  # Pre-LN
hidden_states = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states             # 残差连接
```
输入是 `LayerNorm(上一个 Block 的输出)`。和 `x + self.attn(self.ln1(x))` 完全一样。

**Q: `mlp` 的输入是什么?**

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)  # 第二个 LN
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```
输入是 `LayerNorm(attention 输出 + 残差)`。和 `x + self.ffn(self.ln2(x))` 完全一样。

**Q: 和 nanoGPT 的 `Block` 对比有什么差异?**

结构一样（Pre-LN + 两次残差）。差异在组件内部：RMSNorm vs LayerNorm、SwiGLU vs GELU、RoPE vs 可学习 PE。骨架相同，组件升级。

---

## Trainer.training_step (Day 3-4)

**Q: 一次 `training_step` 的完整流程?**

```python
def training_step(self, model, inputs):
    inputs = self._prepare_inputs(inputs)       # 移到 GPU
    with self.autocast():                        # 混合精度
        loss = self.compute_loss(model, inputs)  # forward + loss
    if self.args.gradient_accumulation_steps > 1:
        loss = loss / self.args.gradient_accumulation_steps  # 梯度累积归一化
    self.accelerator.backward(loss)              # backward
    return loss.detach()
```
和 Week 2 写的训练循环本质一样：forward → loss → backward。多了 autocast 和 gradient accumulation。

**Q: `compute_loss` 是怎么拿到 loss 的?**

```python
def compute_loss(self, model, inputs):
    outputs = model(**inputs)         # forward
    loss = outputs.loss               # 模型自己算的 loss
    return loss
```
HF 的模型 `forward` 在有 `labels` 时自动算 cross_entropy loss 并返回。你 Week 1 的 MiniGPT 也是这么做的：`if targets is not None: loss = F.cross_entropy(...)`.

**Q: gradient accumulation 在哪里处理的?**

```python
# Trainer.training_loop 里
if (step + 1) % self.args.gradient_accumulation_steps == 0:
    self.optimizer.step()      # 累积够了才 step
    self.optimizer.zero_grad() # 清梯度
```
和 Week 2 提炼的 nanoGPT 梯度累积逻辑完全一样。

---

## DataCollatorForLanguageModeling (Day 3-4)

**Q: 怎么自动 pad?**

```python
def __call__(self, features):
    # 找到 batch 内最长序列，其他序列 pad 到同样长度
    batch = self.tokenizer.pad(features, padding=self.padding)
    # 结果: 所有序列等长，pad_token_id 填充
```
 Week 1/2 不需要 pad（char-level/BPE 都是等长截断）。HF 要处理变长输入。

**Q: `labels` 是怎么生成的?**

```python
# mlm=False 时 (causal LM)
batch["labels"] = batch["input_ids"].clone()
# labels 就是 input_ids 的副本，shift 由模型的 loss 计算内部处理
# (模型内部: logits[:, :-1, :] vs labels[:, 1:])
```

**Q: `mlm=False` 时 causal mask 在哪里?**

DataCollator 不生成 causal mask。因果掩码在模型的 attention 层内部处理（`is_causal=True` 传给 SDPA）。DataCollator 只管 pad 和 labels。

---

模型对比和训练对比见 `results/` 目录：
- `results/week3_model_comparison.md` — nanoGPT vs HF 模型实现对比
- `results/week3_trainer_comparison.md` — nanoGPT vs HF Trainer 对比

### 1. Forward 主路径

```python
# nanoGPT — 一个类，~30 行 forward
class GPT(nn.Module):
    def forward(self, idx, targets=None):
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(T))
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
```

```python
# HF Llama — 三层继承，forward ~80 行（含 KV cache 逻辑）
class LlamaForCausalLM(LlamaPreTrainedModel):
    def forward(self, input_ids, past_key_values=None, ...):
        outputs = self.model(input_ids, past_key_values=past_key_values, ...)
        logits = self.lm_head(outputs[0])
        # + loss 计算、KV cache 更新、attention mask 生成
```

结论：数据流完全一样（embedding → blocks → norm → lm_head），HF 多了 KV cache 和 loss 计算的通用处理。

### 2. KV Cache

```python
# nanoGPT — 没有，生成时每步重新算全部 attention
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = self(idx)  # 每步对整个序列重新 forward
```

```python
# HF — past_key_values 缓存历史 K/V
def forward(self, input_ids, past_key_values=None):
    # 第 1 步: 算完整序列的 K/V，存入 past_key_values
    # 第 2 步起: 只算新 token 的 K/V，和历史拼接
    # 生成 1000 token: nanoGPT 算 1000 次 O(N²)，HF 算 1000 次 O(N)
```

结论：KV cache 是推理优化，训练时所有 token 并行算不需要。但部署时必须有，否则生成很慢。

### 3. 位置编码

```python
# nanoGPT — 可学习 embedding，和 token embedding 相加
self.wpe = nn.Embedding(block_size, n_embd)
x = self.wte(idx) + self.wpe(torch.arange(T))
```

```python
# HF Llama — RoPE，在 attention 内部旋转 Q/K
def forward(self, hidden_states, position_ids=None):
    q, k = self.q_proj(hidden_states), self.k_proj(hidden_states)
    cos, sin = self.rotary_emb(q, position_ids)
    q = apply_rotary_pos_emb(q, cos, sin)  # 旋转 Q
    k = apply_rotary_pos_emb(k, cos, sin)  # 旋转 K
```

结论：可学习 PE 受训练长度限制（block_size=256 就只能看 256 token）。RoPE 通过旋转编码相对位置，可以外推到更长序列，现代模型标配。

### 4. MLP

```python
# nanoGPT — GELU，两层线性
class MLP(nn.Module):
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))
    # x → (D → 4D) → GELU → (4D → D)
```

```python
# HF Llama — SwiGLU，三层线性（多一个 gate）
class LlamaMLP(nn.Module):
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    # x → gate(D → D') * up(D → D') → SiLU → down(D' → D)
```

结论：SwiGLU 比 GELU 多一个 gate 投影（参数量多 ~50%），但效果更好，LLaMA 论文证明的。

### 5. 归一化

```python
# nanoGPT — LayerNorm
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
x_hat = (x - mean) / sqrt(var + eps)
return gamma * x_hat + beta
```

```python
# HF Llama — RMSNorm（不减均值，不算 beta）
var = x.pow(2).mean(dim=-1, keepdim=True)
x_hat = x / sqrt(var + eps)
return gamma * x_hat
```

结论：RMSNorm 少了减均值和 beta 参数，快 ~10-15%，效果接近。大模型标配。

### 6. 训练循环

```python
# nanoGPT — 手写 ~200 行
for step in range(max_iters):
    lr = get_lr(step, ...)
    x, y = get_batch(...)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

```python
# HF — 一行
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

结论：Trainer 内部做的事和 nanoGPT 一模一样（forward → loss → backward → clip → step），但额外封装了 logging、eval、save best、gradient accumulation、mixed precision、wandb、early stopping。你写 200 行的训练逻辑，Trainer 用配置项覆盖了。

### 总结

架构完全一样：embedding → N × Block(attn + ffn) → norm → lm_head。

差异不在架构，在工程：
- **RoPE / RMSNorm / SwiGLU / GQA** — 模型层面的改进，效果更好
- **KV cache** — 推理加速，训练不需要
- **Trainer** — 把训练循环从 200 行代码变成配置项
- **DataCollator** — 自动处理变长序列的 pad 和 label

nanoGPT = "最小可训练的 GPT"，HF = "最小可部署的 GPT"。
