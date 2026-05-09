# Week 3 源码阅读笔记模板

## LlamaForCausalLM.forward

```python
# 源码位置: transformers/src/transformers/models/llama/modeling_llama.py
```

### TODO: 逐行注释 (Day 1-2)
- [ ] `input_ids` 怎么变成 embedding?
- [ ] `past_key_values` 是什么? KV cache 怎么工作的?
- [ ] `lm_head` 和 `embed_tokens` 的 weight tying 在哪里?
- [ ] `causal_mask` 在哪里生成的?

### 问题清单
- [ ] 为什么 `LlamaForCausalLM` 的 `forward` 比 nanoGPT 的 `forward` 长这么多?
- [ ] `position_ids` 是自动生成的还是外部传入的?

---

## LlamaModel.forward

### TODO: 逐层追踪 (Day 1-2)
- [ ] `embed_tokens` → 逐层 `LlamaDecoderLayer` → `norm` → `output`
- [ ] 每一层的 hidden_states shape 是什么?

---

## LlamaDecoderLayer

### TODO: 理解两次残差 (Day 1-2)
- [ ] `self_attn` 的输入是什么?
- [ ] `mlp` 的输入是什么?
- [ ] 和 nanoGPT 的 `Block` 对比,有什么差异?

---

## Trainer.training_step (Day 3-4)

```python
# 源码位置: transformers/src/transformers/trainer.py
```

### TODO
- [ ] 一次 `training_step` 的完整流程?
- [ ] `compute_loss` 是怎么拿到 loss 的?
- [ ] gradient accumulation 在哪里处理的?

---

## DataCollatorForLanguageModeling (Day 3-4)

### TODO
- [ ] 怎么自动 pad?
- [ ] `labels` 是怎么生成的?
- [ ] `mlm=False` 时 causal mask 在哪里?

---

## nanoGPT vs HF 实现对比 (Day 2 末尾)

| 维度 | nanoGPT | HF Transformers |
|------|---------|-----------------|
| 接口抽象 | TODO | TODO |
| KV cache | TODO | TODO |
| weight tying | TODO | TODO |
| 初始化 | TODO | TODO |
| 位置编码 | TODO | TODO |
