# Week 5: SFT 严肃实战 — Chat Template + Loss Masking

> 目标: 理解 SFT 的每个细节,掌握 chat template 和 loss masking。
> 预计时间: 14-20 小时

> **上周回顾**: Week 4 你理解了 LoRA 的数学 (SVD 视角) 和实现 (手写 + PEFT 源码)。LoRA 解决的是**硬件问题** — 用更少显存训练。这周解决的是**数据问题** — 怎么正确地喂训练数据。
>
> **为什么学这周**: SFT 是让 base model 变成 usable assistant 的关键步骤。如果你不理解 loss masking (只在 assistant 回复上算 loss),你的模型会浪费大量梯度去学习"怎么重复用户的 prompt"。这个 bug 很隐蔽 — 训练 loss 正常下降,但模型效果很差。
>
> **思考锚点**: "SFT 训练时,如果 labels = input_ids (不做 masking),模型在学什么? 如果做 masking,模型又只学什么?"

---

## Day 1-2: Chat Template 深度

> **思考**: 如果用 Llama-3 的 template 去训练 Qwen 模型,会发生什么? 为什么 template 必须匹配?

### 做什么
1. 研究 3 种主流 template:
   - **Qwen (ChatML)**: `<|im_start|>system\n{system}<|im_end|>\n...`
   - **Llama-3**: `<|start_header_id|>system<|end_header_id|>\n\n{system}...`
   - **Mistral**: `<s>[INST] {user} [/INST] {response}</s>`
2. 实验: 同一条对话用 3 种 template tokenize,对比 token 数
3. 实验: 用 Llama-3 template 训练 Qwen 模型 → 观察效果下降

### 跑
```bash
python phase0/week5/chat_template_compare.py
```

### 交付物
- `phase0/results/week5_template_comparison.json` — 3 种 template 对比
- `phase0/notes/week5_chat_template.md` — template 理解笔记

---

## Day 3-4: Loss Masking

> **思考**: `ignore_index=-100` 这个魔数是什么意思? PyTorch 的 CrossEntropyLoss 怎么处理它?

### 做什么
1. 理解核心 trick: SFT 只在 assistant response 的 token 上计算 loss
2. 手写 `mask_labels` 函数
3. 对比实验: 不 masking vs masking → 评估效果差异
4. 理解 multi-turn: 每个 assistant turn 都要 mask

### 具体步骤
Loss masking 的实现思路:
```
原始 input_ids:  [system tokens] [user tokens] [assistant tokens]
labels (无masking): [system tokens] [user tokens] [assistant tokens]
labels (有masking): [-100, -100, ...] [-100, -100, ...] [assistant tokens]
                                       ↑ 只在这里算 loss
```

在 `loss_masking.py` 中你需要:
1. 找到 `<|im_start|>assistant\n` 的 token 序列
2. 标记它之后的所有 token 为 "需要计算 loss"
3. 其他位置的 label 设为 -100

### 跑
```bash
python phase0/week5/loss_masking.py
```

### 交付物
- `phase0/week5/loss_masking.py` 中的 `mask_labels` 实现
- 对比实验记录

---

## Day 5-6: 数据质量 vs 数量实验

> **思考**: LIMA 论文说 "1000 条高质量数据 > 50000 条低质量数据"。直觉上为什么? (提示: 模型从噪声中学到的也是噪声)

### 做什么
1. 准备 3 份数据:
   - A: 500 条高质量 (人工审核)
   - B: 2000 条中等质量 (自动清洗)
   - C: 5000 条低质量 (原始爬取)
2. 用 QLoRA 在 Qwen2.5-3B 上分别训练
3. 在领域测试集上评估
4. 复现 LIMA 核心结论: 数据质量 > 数量

### 交付物
- `phase0/results/week5_quality_vs_quantity.md` — 实验报告

---

## Day 7: SFT 最佳实践整理

### 做什么
整理个人 **SFT Checklist**:
- 数据: 质量 > 数量,500 条好的 > 5000 条差的
- Learning rate: `2e-4` (QLoRA) / `5e-5` (全量)
- Epochs: `1-3`,多了过拟合
- Batch size: 尽可能大,用 gradient accumulation 模拟
- LoRA rank: `8` (小模型) 或 `16` (大模型)
- LoRA alpha: 通常 = rank 或 2x rank
- Target modules: `["q_proj", "v_proj"]` 最少,加更多更好

### 交付物
- `phase0/notes/week5_sft_checklist.md`

---

## 自测题

1. **ChatML 中 `<|im_start|>` 和 `<|im_end|>` 各是什么作用?** 如果 `<|im_end|>` 丢失了会怎样?
2. **Loss masking 不做的话,模型会学到什么不该学的东西?**
3. **为什么 QLoRA 的 learning rate (2e-4) 比全量微调 (5e-5) 大?**

> 答案: 1) `<|im_start|>` 标记角色开始, `<|im_end|>` 标记角色结束。丢失 end token 会让模型无法区分不同角色的边界,可能把 system prompt 和 user input 混在一起。2) 模型会学习生成用户的 prompt — 在 inference 时,模型可能开始重复用户的问题而不是回答。3) LoRA 只训练极少量参数 (0.1%),需要更大的 LR 才能在有限的参数空间内学到足够的信号;全量微调参数多,小 LR 就够了。

---

## 验收清单

- [ ] 3 种 template 对比实验
- [ ] loss masking 手写实现 + 对比实验
- [ ] 数据质量 vs 数量实验报告
- [ ] 个人 SFT Checklist
- [ ] 自测题能回答 2/3 以上
