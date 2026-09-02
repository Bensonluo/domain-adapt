# Week 5: SFT 严肃实战 — Chat Template + Loss Masking

> 目标: 理解 SFT 的每个细节,掌握 chat template 和 loss masking。
> 预计时间: 14-20 小时
>
> **本周交付**：已完成 template、masking 与 SFT 实现；训练效果消融尚未运行。详见 [CORRECTIONS.md](CORRECTIONS.md)。

> **前后衔接**: Week 4 讨论 LoRA 的数学视角与实现；本周转向 SFT 的数据格式与训练目标，检查 chat template、loss masking 和数据质量如何影响训练。
>
> **为什么学这周**: SFT 是让 base model 变成 usable assistant 的关键步骤。assistant-only loss 让优化目标更贴近期望回答，避免长 prompt 主导 token loss；不 masking 不必然导致 prompt repetition，实际影响需要受控实验测量。
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

`loss_masking.py` 的实现步骤：
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

## Day 5-6: 数据质量与数量解耦实验

> **思考**: 数据质量与数量可能交互；若两者同时变化，就无法判断效果来自哪一个因素。

### 做什么
1. 质量主效应：固定 500 条及相近 assistant token，总体主题/难度/长度匹配，比较高质量与低质量。
2. 数量主效应：从同一质量池抽取 500/2000/5000 条。
3. 固定模型、模板、LoRA、优化器和评估集；至少 3 个 seed。
4. 分别报告固定 epoch 与固定优化 token/step，避免把算力增加误认成数据多样性收益。
5. 在新 blind test 上报告置信区间；实验前只能把“质量重要”写成假设。

### 交付物
- `phase0/results/week5_quality_vs_quantity.md` — 实验报告

---

## Day 7: SFT 最佳实践整理

### 做什么
整理个人 **SFT Checklist**:
- 数据: 质量和数量分开控制；先审计去重、正确性、覆盖与难度，再通过实验确定投入优先级
- Learning rate: `2e-4` (QLoRA) / `5e-5` (全量)
- Epochs: `1-3`,多了过拟合
- Batch size: 尽可能大,用 gradient accumulation 模拟
- LoRA rank: `8` (小模型) 或 `16` (大模型)
- LoRA alpha: 通常 = rank 或 2x rank
- Target modules: 从 q/v 或 attention projections 起步；是否扩展到更多模块由参数预算和消融决定，不是越多越好

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

- [x] 3 种 template 的 tokenization 对比
- [ ] template 错配的训练效果对照
- [x] loss masking 标签实现与 token 级检查
- [ ] masking vs unmasked 训练效果对照
- [ ] 数据质量与数量解耦实验报告
- [x] 个人 SFT Checklist
- [x] 自测题能回答 2/3 以上

---

## 成果

**Chat Template 对比** — [chat_template_compare.py](chat_template_compare.py) 对比 Qwen/ChatML、Llama-3、Mistral 三种 template 对同一条医疗对话的 tokenize 结果，分析 token 数差异、system 消息处理方式、模板错配的后果。见 [chat_template.ipynb](chat_template.ipynb) 的交互式对比。

**Loss Masking 实现** — [loss_masking.py](loss_masking.py) 保留每个 assistant turn 的内容及结束 token，prompt/角色前缀设为 -100。当前完成的是目标构造与 token 级检查，尚未完成下游效果对照。

**完整 SFT 训练脚本** — [sft_trainer.py](sft_trainer.py) 整合 chat template + loss masking + QLoRA，支持 Qwen2.5-3B-Instruct 的完整 SFT 流程。默认起始配置为 LoRA r=16、alpha=32、target_modules=q/k/v/o_proj 和 NF4 量化，不代表已验证最优值。

训练脚本要求显式提供预先按 prompt/实体/来源分组得到的 `--eval_data`，不再在训练脚本中随机按行拆分 dev，以避免组级近重复泄漏；每个 epoch 计算 dev loss，并按 `eval_loss` 恢复最佳 checkpoint。

**SFT Checklist** — [sft_checklist.md](sft_checklist.md) 整理数据、模板、masking 和训练检查项。超参均为起始范围，不是跨模型通用最优值。
