# Week 1: PyTorch 底层 + Attention 手写 + 训练循环

> 目标: 理解 autograd,手写 multi-head attention,组装完整 Transformer Block,跑通训练循环。
> 预计时间: 14-20 小时。

> **为什么学这周**: Autograd 是所有深度学习的基石 — 不知道梯度怎么算就无法理解训练为什么能工作。Attention 是 Transformer 的核心,也是你未来做 LoRA/蒸馏/RLHF 时最常打交道的模块。这两者不理解,后面所有东西都是空中楼阁。
>
> **思考锚点** (贯穿本周): "一个计算图中的梯度,究竟是怎么从 loss 一路流回每个参数的?"

---

## 目录

| 文件 | 内容 |
|------|------|
| `day1_2_autograd.py` | autograd 实验 + manual backward 对照 |
| `day3_4_attention.py` | 手写 scaled dot-product attention + multi-head |
| `day5_7_transformer.py` | LayerNorm + FFN + Block + MiniGPT |
| `train_toy.py` | 在 tiny_shakespeare 上跑训练循环 |

---

## Day 1-2: PyTorch autograd 底层

> **思考**: 为什么不用有限差分 (finite differences) 算梯度? 提示: 参数量 1M 的模型,有限差分需要 1M 次 forward pass,而 backward 只要 1 次。

### 做什么
1. 看 Karpathy micrograd 前 60 分钟: [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
   - **主动观看**: 每看到一个新概念暂停,尝试自己预测下一步实现。重点记录: Value 类怎么构建计算图? backward 怎么拓扑排序?
2. 阅读 PyTorch autograd 教程: [官方文档](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
3. 打开 `day1_2_autograd.py`,按 TODO 填完 `backward_manual` 的 4 步
4. 运行脚本,确认 manual 和 auto 梯度误差 < 1e-6

### 跑
```bash
source .venv/bin/activate
python phase0/week1/day1_2_autograd.py
```

### 交付物
- `backward_manual` 完整实现(代码)
- 梯度对比通过截图 → 存入 `phase0/results/autograd_check.png`
- 手写推导 `d_loss/d_W1` 的链式法则照片 → 存入 `phase0/notes/week1_d1.pdf`

---

## Day 3-4: Multi-Head Attention 手写

> **思考**: Scaled Dot-Product Attention 为什么要除以 √d_k? 如果不除,当 d_k=64 时 Q·K 的方差是多少? (提示: 假设 Q,K 独立标准正态,点积方差 = d_k)

### 做什么
1. 看 Karpathy "Let's build GPT" 前 30 分钟到 attention: [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
   - **主动观看**: 在他写 attention 之前暂停,尝试自己写出 Q/K/V 的 shape 变换。
2. 打开 `day3_4_attention.py`,按 TODO 填完 `MultiHeadAttention.forward` 的 5 步
3. 运行一致性测试,确认和 `F.scaled_dot_product_attention` 一致

### 跑
```bash
python phase0/week1/day3_4_attention.py
```

### 交付物
- `MultiHeadAttention` 完整实现(代码)
- 白板照片: Q/K/V shape 流 → `phase0/notes/week1_attention_shapes.jpg`
- 一致性测试通过截图

---

## Day 5-7: Transformer Block + 训练循环

> **思考**: Token embedding 和 position embedding 为什么是相加而不是拼接? (提示: 拼接会让维度翻倍,而相加在数学上等价于投影到同一子空间后的组合)

### 做什么
1. 打开 `day5_7_transformer.py`,按 TODO 填完 `MyLayerNorm` 和 `FFN.forward`
2. 确认 `smoke_test` 通过: 参数量 + 初始 loss 接近 log(vocab_size)
3. 运行 `train_toy.py`,看到 loss 下降
4. 尝试不同 `temperature` 生成文本,观察效果差异

### 跑
```bash
# 模块测试
python phase0/week1/day5_7_transformer.py

# 训练 (Mac CPU 约 10-15 分钟, MPS 约 3-5 分钟)
python phase0/week1/train_toy.py
```

### 交付物
- `MyLayerNorm` + `FFN.forward` 完整实现(代码)
- `loss_curve.png` (train_toy.py 自动生成)
- `sample.txt` 生成文本(训练后自动生成)
- 手写笔记: Pre-LN vs Post-LN 对比 → `phase0/notes/week1_transformer_notes.md`

---

## Week 1 额外任务

### lm-eval 基线
```bash
# 在基座模型上跑医疗相关 MMLU 子集,记录分数
python phase0/utils/eval_baseline.py
```
产出: `phase0/results/baseline_qwen25_3b.json`

> 3B 模型在 Mac 上推理慢(CPU 可能 30+ 分钟跑完)。如果太慢可以先跳过,Week 8 有完整评估方法论。

---

## 自测题

做完本周所有任务后,尝试不查资料回答:

1. **`requires_grad=True` 的张量参与运算后,新张量的 `grad_fn` 是什么?** 它为什么重要?
2. **Multi-head attention 的 "multi-head" 在做什么?** (a) 多个独立的注意力函数 (b) 在不同子空间并行计算注意力然后合并
3. **Pre-LN (`x + Attn(LN(x))`) 和 Post-LN (`LN(x + Attn(x))`) 的核心区别是什么?** 为什么深层网络用 Pre-LN 更稳定?

> 答案: 1) 指向创建该张量的运算,构成计算图的边;autograd 通过它回溯梯度。2) (b) — 把 d_model 拆成 h 个 d_k 子空间,各自算注意力再 concat,让模型关注不同位置的不同表征子空间。3) Pre-LN 的残差路径上没有 LayerNorm,梯度可以直接走 skip connection 流回浅层,深层网络更稳定。

---

## 验收清单

- [ ] `day1_2_autograd.py` 梯度对比通过
- [ ] `day3_4_attention.py` 一致性测试通过
- [ ] `day5_7_transformer.py` smoke_test 通过
- [ ] `train_toy.py` loss 从 ~4.5 降到 < 2.0
- [ ] 手写推导照片/白板存档
- [ ] 自测题能回答 2/3 以上
- [ ] (可选) lm-eval 基线跑通

---

## 常见问题

**Q: MPS 上跑 loss 有 NaN?**
A: 早期 PyTorch MPS backend 有 GELU 精度问题,试试 `torch.backends.mps.synchronize()` 或升级 PyTorch 到 2.4+。如果是 loss 为 NaN,先检查 `MyLayerNorm` 的 `eps` 是否被设为 0。

**Q: train_toy.py 在 CPU 上太慢?**
A: 把 `d_model=128` 降到 `64`、`n_layers=4` 降到 `2`、`max_iters=2000` 降到 `500`,先跑一个 2 分钟版本验证流程通。

**Q: 生成的文本完全不像英文?**
A: 正常。字符级 tokenizer + 2000 steps + 只有 ~0.1M 参数的 toy 模型,只能学到字母组合规律,学不到语义。这是 OK 的,目标是理解训练循环,不是产出好模型。
