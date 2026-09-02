# Week 3: HuggingFace Transformers 源码 + 全量微调

> 目标: 从"用 API"到"理解 API 下面发生了什么"。做一次全量微调(不用 PEFT)。
> 预计时间: 14-20 小时
>
> **本周交付**：已完成源码阅读、框架比较与训练脚本；full FT 资源实测尚未运行。详见 [CORRECTIONS.md](CORRECTIONS.md)。

> **前后衔接**: Week 2 阅读 nanoGPT 的简洁实现；本周对比 HuggingFace Transformers，梳理框架增加了哪些工程能力。
>
> **学习重点**: 阅读 Trainer、模型 forward 和数据管线，理解配置与训练行为之间的关系，为 Week 4 的 LoRA 和 Week 5–6 的 QLoRA SFT 提供实现依据。
>
> **思考锚点**: "HuggingFace 的 Trainer 和 nanoGPT 的训练循环,本质做的是同一件事。差距在于通用性 — 它多处理了哪些 edge case?"

---

## Day 1-2: Transformers 源码 — Model 部分

> **思考**: HF 的 `LlamaForCausalLM.forward` 与 Week 1 的 `MiniGPT.forward` 在流程上有哪些共同点和差异? (提示: KV cache)

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

> **思考**: Trainer 的 `training_step` 做了什么? 与 Week 2 的 `train.py` 训练循环相比，多了哪些步骤?

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

> **思考**: 全量微调 1.5B 模型需要多少显存? FP32 AdamW 的权重、梯度、m、v 仅参数相关部分约为 4 个 FP32 张量；实际显存还包含激活、临时 buffer 和框架开销。为什么 Week 4 要学 LoRA?

### 做什么
1. 加载 `Qwen/Qwen2.5-1.5B-Instruct`
2. 准备 500-2000 条领域指令数据(JSONL 格式)
3. 用 `Trainer` 做全量 SFT(不用 PEFT/LoRA,先理解全量微调的显存消耗)
4. 记录: 训练时间、GPU 显存峰值、loss 曲线

### 跑 (GPU 服务器)
```bash
# 在 GPU 服务器上
cd /root/workspace/domain-adapt/phase0/week3
python train_full_ft.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data /path/to/domain_data.jsonl \
    --eval_data /path/to/domain_dev.jsonl \
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

> 答案: 1) 自回归生成时每步只多一个 token,但 attention 要看所有历史。KV cache 把历史 token 的 K/V 存下来,避免重复计算。训练时所有 token 并行处理,通常不使用生成式 KV cache。2) FP32 AdamW 的简化下界约为 1.5B × 4 个张量 × 4 bytes = 24 GB，仅计权重、梯度、m、v；实际还包含激活、临时 buffer，混合精度实现还可能保留 master weights。3) collator 负责动态 padding、padding label masking，并可从 input_ids 构造 labels；causal LM 的 logits/labels shift 通常在模型 loss 内部完成，不是 collator 预先右移。

---

## 验收清单

- [ ] `LlamaForCausalLM.forward` 逐行注释
- [ ] `Trainer.training_step` 理解流程
- [ ] 全量微调跑通,记录显存峰值
- [ ] 代码结构图完成
- [ ] 自测题能回答 2/3 以上
