# Week 3: nanoGPT vs HF 模型实现对比

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
