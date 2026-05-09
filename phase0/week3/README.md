# Week 3: HuggingFace Transformers 源码 + 全量微调

> 目标: 从"用 API"到"理解 API 下面发生了什么"。做一次全量微调(不用 PEFT)。
> 预计时间: 14-20 小时

> **上周回顾**: Week 2 你看了 nanoGPT — 一个研究者写的简洁实现。这周看 HuggingFace Transformers — 工业界标准。你要对比两者的差距,理解"工程化"到底加了什么。
>
> **为什么学这周**: 你以后所有训练实验都用 HuggingFace 生态 (Trainer + PEFT + datasets)。如果不懂它底层在做什么,出了问题只能猜。读懂源码之后,Week 4 加 LoRA、Week 5-6 做 QLoRA SFT 就是加几行配置,而不是黑魔法。
>
> **思考锚点**: "HuggingFace 的 Trainer 和 nanoGPT 的训练循环,本质做的是同一件事。差距在于通用性 — 它多处理了哪些 edge case?"

---

## Day 1-2: Transformers 源码 — Model 部分

> **思考**: HF 的 `LlamaForCausalLM.forward` 和你 Week 1 写的 `MiniGPT.forward` 流程几乎一样。找找最大的差异在哪里? (提示: KV cache)

### 做什么
1. 源码安装: `git clone https://github.com/huggingface/transformers && pip install -e .`
2. 阅读 `LlamaForCausalLM` 的 `forward` 方法(~50 行):
   - 输入 `input_ids` → `embed_tokens` → 过 `self.model`(Transformer layers) → `lm_head`
   - `past_key_values` 的 KV cache 机制(加速推理)
3. 阅读 `LlamaModel` 的 `forward`:
   - `embed_tokens` → 逐层过 `LlamaDecoderLayer` → `norm` → `output`
4. 阅读 `LlamaDecoderLayer`:
   - `self_attn` → `mlp` → 两次残差连接
5. 对比 nanoGPT 和 HF 的实现差异(接口抽象程度、KV cache、weight tying)

### 阅读方法
不要通读 — 带着问题跳读:
1. 先看 `forward` 的输入输出签名
2. 跟踪 `input_ids` 从进入到输出的完整路径
3. 遇到不懂的参数先跳过,只看主路径

### 交付物
- `phase0/notes/week3_hf_modeling.md` — 源码阅读笔记
- `phase0/notes/week3_nano_vs_hf.md` — nanoGPT vs HF 实现对比

---

## Day 3-4: Transformers 源码 — Trainer 部分

> **思考**: Trainer 的 `training_step` 做了什么? 和你 Week 2 的 `train.py` 训练循环对比,多了哪些步骤?

### 做什么
1. 阅读 `Trainer.training_step`: 一次迭代 = forward → loss → backward → 返回 loss dict
2. 阅读 `Trainer.compute_loss`: 怎么从 model output 拿到 loss
3. 理解 `DataCollatorForLanguageModeling`: 怎么自动 pad + mask
4. 理解 `TrainingArguments` 和 `Trainer` 的职责分工

### 交付物
- `phase0/notes/week3_hf_trainer.md` — Trainer 源码笔记
- `phase0/notes/week3_data_collator.md` — DataCollator 理解

---

## Day 5-6: 全量微调实验(不用 PEFT)

> **思考**: 全量微调 1.5B 模型需要多少显存? (提示: AdamW 需要 2 个状态矩阵,每个参数 = 模型权重 + 梯度 + m + v = 4x 参数量的显存) 为什么 Week 4 要学 LoRA? 因为你这次会亲眼看到显存不够用。

### 做什么
1. 加载 `Qwen/Qwen2.5-1.5B-Instruct`
2. 准备 500-2000 条领域指令数据(JSONL 格式)
3. 用 `Trainer` 做全量 SFT(不用 PEFT/LoRA,先理解全量微调的显存消耗)
4. 记录: 训练时间、GPU 显存峰值、loss 曲线

### 跑 (GPU 服务器)
```bash
# 在 GPU 服务器上
cd /root/workspace/growing-big/phase0/week3
python train_full_ft.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data /path/to/domain_data.jsonl \
    --output_dir ./results_full_ft \
    --epochs 3 \
    --batch_size 2 \
    --lr 5e-5
```

### 交付物
- `phase0/results/week3_full_ft_log.txt` — 训练日志
- `phase0/results/week3_memory.txt` — 显存峰值记录
- `phase0/results/week3_loss_curve.png` — loss 曲线

---

## Day 7: 复盘

### 做什么
- 画一张 HF Transformers 代码结构图(从 model 到 trainer 到 data collator)
- 标记: 哪些地方还不懂

### 交付物
- `phase0/notes/week3_architecture_diagram.md` — 代码结构图
- `phase0/notes/week3_gaps.md` — 不懂的地方清单

---

## 自测题

1. **KV cache 解决了什么问题?** 为什么推理时用它而训练时不用?
2. **全量微调 1.5B 模型,AdamW 的显存开销大约是多少?** (FP32 训练)
3. **`DataCollatorForLanguageModeling` 做了哪两件事?**

> 答案: 1) 自回归生成时每步只多一个 token,但 attention 要看所有历史。KV cache 把历史 token 的 K/V 存下来,避免重复计算。训练时所有 token 并行处理,不需要。2) 约 1.5B × 4 (权重 + 梯度 + m + v) × 4 bytes = ~24 GB (仅参数,不含激活)。3) (a) 动态 padding 到 batch 内最长序列; (b) 对 causal LM 自动创建 labels (右移一位的 input_ids)。

---

## 验收清单

- [ ] `LlamaForCausalLM.forward` 逐行注释
- [ ] `Trainer.training_step` 理解流程
- [ ] 全量微调跑通,记录显存峰值
- [ ] 代码结构图完成
- [ ] 自测题能回答 2/3 以上
