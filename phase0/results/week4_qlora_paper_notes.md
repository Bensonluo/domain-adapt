# QLoRA 论文要点

## 1. 核心思想

```
QLoRA = 量化基座 (NF4) + LoRA 适配器 (BF16)
```

基座模型用 4bit 存储，LoRA 适配器保持 16bit 精度训练。效果和 16bit 全量微调几乎一样，显存降到 1/4。

论文的 claim：**在 48GB 显存上微调 65B 模型，效果不输全量微调。**

## 2. NF4 — 4-bit Normal Float

**问题：为什么不用 INT4？**

```
INT4 均匀量化:  16 个等距区间 [-8, -7, ..., 7]
LLM 权重分布:   近似正态分布 N(0, σ²)，大部分值集中在 0 附近

INT4 的问题:    大权重和小权重用同样大的区间 → 小权重精度差
                但 LLM 90%+ 的权重都在 [-2σ, 2σ] 范围内
```

**NF4 的做法：**

```
1. 假设权重服从正态分布 N(0, 1)
2. 把正态分布的 CDF 等分成 16 个区间（每个区间概率 = 1/16）
3. 取每个区间的中位数作为量化值

结果: 16 个量化值在 0 附近密集，两端稀疏
     正好匹配 LLM 权重的分布特征
     权重密集的地方精度高，权重稀疏的地方精度低
```

实际操作：
```python
# 量化
q_value = nf4_quantize(weight)      # FP32 → NF4 (0.5 bytes)
# 反量化（计算时临时还原）
weight_approx = nf4_dequantize(q_value, scale)  # NF4 → BF16
# 反量化后的权重参与矩阵乘，用完即丢
```

## 3. Double Quantization — 对量化常数再量化

```
普通量化: 每组 64 个权重共享一个 scale（FP32）
         scale 本身占 32 bits / 64 weights = 0.5 bits/weight 的额外开销

Double Quantization:
  第一层: 权重 → NF4 (0.5 bytes/param)
  第二层: scale → FP8 (1 byte) + 第二组 scale (FP32)
         额外开销从 0.5 bits/weight 降到 ~0.127 bits/weight

省的不多（每 1B 参数省 ~47 MB），但对 65B 模型就是 ~3 GB
```

## 4. Paged Optimizer — 解决显存峰值问题

```
问题: Adam 优化器状态 (m, v) 占大量显存
      某些步骤（gradient checkpointing 重新计算时）显存会突增
      显存峰值 → OOM

解决: 把优化器状态放在 CPU 内存（便宜）
      需要时用统一内存自动 paging 到 GPU
      NVIDIA 统一内存: CPU/GPU 共享地址空间，自动换页
```

实际效果：65B 模型训练时峰值显存从 ~48GB 降到 ~48GB 但不再 OOM（消除了尖峰）。

## 5. 数据流：QLoRA 训练的一步

```
1. 从磁盘加载 NF4 量化权重 → 反量化到 BF16 → 拼上 LoRA 适配器输出
2. Forward: y = W_dequant @ x + (α/r) × B @ A @ x
3. 计算 loss，反向传播
4. 梯度只更新 LoRA 的 A 和 B（BF16），基座权重不动
5. A/B 的 Adam 优化器状态存在 CPU (Paged Optimizer)
```

关键：反量化是临时操作，不常驻显存。每个 batch 只在 forward 时按需反量化。

## 6. QLoRA vs 16-bit 全量微调（论文 Figure 1）

```
实验: LLaMA 65B, GSM8K (数学推理), MMLU (多任务)

| 方法              | 可训练参数 | 显存  | MMLU  | GSM8K |
|-------------------|-----------|-------|-------|-------|
| 16-bit 全量微调   | 65B       | ~780GB| 基准  | 基准  |
| 16-bit LoRA       | ~20M      | ~260GB| -0.1% | -0.3% |
| QLoRA (NF4)       | ~20M      | ~48GB | -0.0% | -0.2% |

结论: QLoRA 和 16-bit 全量微调效果几乎无差别，但显存是 1/16
```

## 7. 三大创新的关系

```
NF4                 → 基座模型从 16bit 压到 4bit（省 3/4 显存）
Double Quantization → 量化常数的额外开销再砍 75%
Paged Optimizer     → 优化器状态卸到 CPU，消除显存峰值

三者叠加: 让 65B 模型在 48GB 显存上可训练
```

## 8. 你实际用 QLoRA 时的配置

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化
    bnb_4bit_use_double_quant=True,      # Double Quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时用 BF16
)

# LoRA 配置
lora_config = LoraConfig(
    r=8,                                  # rank
    lora_alpha=16,                        # alpha = 2 × rank
    target_modules=["q_proj", "v_proj"],  # 只适配 q 和 v
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
# 可训练参数: ~4M (7B 的 0.06%)
```
