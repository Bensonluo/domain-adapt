# Week 4: PEFT / LoRA 源码深度 + 手写 minimal LoRA

> 目标: 从论文数学到代码实现,彻底理解 LoRA 和 QLoRA。
> 预计时间: 14-20 小时

> **上周回顾**: Week 3 你做了全量微调 — 亲身体验了 1.5B 模型吃掉几十 GB 显存。这周学 LoRA: 只训练 0.1% 的参数,显存降到 1/10,效果几乎不差。
>
> **为什么学这周**: LoRA 是你未来所有训练实验的基础方法。你的方向是 domain adaptation — 需要频繁地在不同领域数据上微调模型,LoRA 让这件事在单卡 4090 上可行。不理解 LoRA 的数学原理(rank 选择、alpha 含义),就只能照抄别人的配置。
>
> **思考锚点**: "LoRA 初始化时 B=0 保证了什么? 如果 B 也随机初始化会怎样?"

---

## Day 1-2: LoRA 论文精读

> **思考**: 论文说只适配 q_proj 和 v_proj 就够了 (Table 4)。为什么不动 k_proj 和 output_proj? 什么情况下你会想动更多模块?

### 做什么
1. 精读 LoRA 论文 (Section 4 + Appendix)
2. 手写推导: `W_new = W_0 + α/r * B @ A`
3. 理解 SVD 视角: 预训练权重的变化矩阵 ΔW 是低秩的
4. 理解为什么只适配 attention 的 q/v (论文 Table 4 的 ablation)
5. 理解 alpha 参数: `scaling = alpha / rank`

### 论文阅读方法
不要从头到尾读 — 先读 Section 4 (实验) 和 Figure 2 (架构图),有直观理解后再回看 Section 3 (数学推导)。

### 资源
- LoRA 论文: https://arxiv.org/abs/2106.09685
- PEFT LoRA 源码: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py

### 交付物
- `phase0/notes/week4_lora_paper.md` — 精读笔记
- 手写 SVD 推导照片

---

## Day 3-4: QLoRA 论文精读

> **思考**: NF4 (4-bit Normal Float) 为什么比 INT4 更适合量化 LLM 权重? (提示: LLM 权重分布是什么形状?)

### 做什么
1. 精读 QLoRA 论文 (Section 2 + Figure 1)
2. 理解 NF4 (4-bit Normal Float)
3. 理解 Double Quantization
4. 理解 Paged Optimizer
5. 对比 QLoRA 和 16-bit 全量微调 (Figure 1)

### 资源
- QLoRA 论文: https://arxiv.org/abs/2305.14314
- PEFT LoRA bnb 源码: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py

### 交付物
- `phase0/notes/week4_qlora_paper.md` — 精读笔记

---

## Day 5-6: 手写 minimal LoRA

> **思考**: GPT-2 的 attention 层用的是 `Conv1D` 而不是 `nn.Linear`。它们的 weight shape 有什么不同? 为什么 `inject_lora` 要同时检查这两种类型?

### 做什么
1. 打开 `phase0/week4/lora_from_scratch.py`,按 TODO 实现
   - **LoRALinear.forward**: `original(x) + scaling * x @ A^T @ B^T`
   - **inject_lora**: 遍历 named_modules,匹配 target_modules,替换为 LoRALinear
2. 注入到 GPT-2 124M 的 attention 层 (target: `c_attn`)
3. 跑 toy 实验: 对比手写 LoRA 和 PEFT 库 LoRA 的 loss 下降曲线
4. 打开 `lora.ipynb`,看 SVD 可视化,理解 rank 选择

### 关键提醒
- GPT-2 的 `c_attn` 是 `Conv1D`,weight shape = `[d_in, d_out]` (和 nn.Linear 的 `[d_out, d_in]` 相反)
- `inject_lora` 需要用 `name.rsplit('.', 1)` 拆出 parent 和 child_name,然后用 `setattr` 替换

### 跑
```bash
source .venv/bin/activate
python phase0/week4/lora_from_scratch.py
python phase0/week4/compare_lora.py
```

### 交付物
- 手写 LoRA 实现 (代码)
- toy 对比实验记录 (loss 曲线)
- SVD 视角 rank 选择推导

---

## Day 7: PEFT 源码阅读

> **思考**: 官方 PEFT 的 `LoraLinear` 实现和你的手写版有什么不同? (提示: 看 `merge_and_unload` 方法和 `scaling` 的处理)

### 做什么
1. 阅读 `peft/tuners/lora/layer.py` (~300 行)
2. 重点看 `Linear` 类的 `forward`
3. 理解 `target_modules` 怎么匹配
4. 对比手写实现和官方实现的差异

### 交付物
- `phase0/notes/week4_peft_source.md` — 源码对比笔记

---

## 自测题

1. **LoRA 的 rank 从 8 增加到 16,可训练参数量怎么变?** 推理时的计算量变不变?
2. **`prepare_model_for_kbit_training` 做了什么?** 为什么 QLoRA 需要这一步?
3. **NF4 量化为什么比普通 INT4 更好?**

> 答案: 1) 可训练参数量大约翻倍 (每个 LoRA 层: A 从 r×d 变成 2r×d, B 从 d×r 变成 d×2r)。推理时 LoRA 已经 merge 回权重,没有额外计算。2) 它把量化模型中需要梯度计算的层转为 float32,并冻结不需要的层,确保 QLoRA 训练的数值稳定性。3) LLM 权重近似正态分布,NF4 的量化分位数是按正态分布设计的,在权重密集的区域有更高的量化精度。

---

## 验收清单

- [ ] LoRA + QLoRA 论文精读笔记
- [ ] 手写 LoRA 实现 (GitHub 提交)
- [ ] toy 对比实验记录 (手写 vs PEFT)
- [ ] SVD 视角 rank 选择推导
- [ ] 自测题能回答 2/3 以上
