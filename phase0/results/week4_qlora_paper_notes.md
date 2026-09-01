# QLoRA 论文要点

## 1. 核心思想

```
QLoRA = 量化基座 (NF4) + LoRA 适配器 (BF16)
```

基座模型以 4bit 量化形式保存，计算时按配置反量化到计算 dtype，梯度只更新 LoRA 参数。显存收益与模型、序列长度、batch、checkpointing 和实现共同相关，不能固定概括为“1/4”。

论文展示了在单张 48GB GPU 上对 65B 模型进行 QLoRA 微调，并在其选定的数据与评估上取得有竞争力的结果；这不等于已经证明 33B/65B 在所有任务上与同条件 16-bit full FT 等价。

## 2. NF4 — 4-bit Normal Float

**问题：为什么不用 INT4？**

```
INT4 均匀量化:  16 个等距区间 [-8, -7, ..., 7]
LLM 权重分布:   近似正态分布 N(0, σ²)，大部分值集中在 0 附近

INT4 的问题:    大权重和小权重用同样大的区间 → 对近零高密度区域的表示效率较低
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

解决: 使用 NVIDIA unified memory，在内存压力出现时让优化器状态分页到 CPU，
      需要时再传回 GPU；这不是“优化器状态始终全部放在 CPU”。
```

作用边界：降低偶发显存峰值导致 OOM 的风险；具体峰值和传输开销必须由运行日志测量，不能从方法描述推导固定数字。

## 5. 数据流：QLoRA 训练的一步

```
1. 从磁盘加载 NF4 量化权重 → 反量化到 BF16 → 拼上 LoRA 适配器输出
2. Forward: y = W_dequant @ x + (α/r) × B @ A @ x
3. 计算 loss，反向传播
4. 梯度只更新 LoRA 的 A 和 B（BF16），基座权重不动
5. A/B 的优化器状态可在显存压力下由 paged optimizer 借助统一内存分页
```

关键：冻结的量化权重不接收梯度；反量化与矩阵乘的具体 buffer 生命周期依赖 bitsandbytes/kernel 实现，不能仅凭概念图断言峰值显存。

## 6. QLoRA 与 16-bit 训练的证据边界

此前把一组 65B/MMLU/GSM8K 数字标成“论文 Figure 1”，没有可核验来源，现已撤回。Figure 1 不能支撑“QLoRA 普遍等同 16-bit full FT、显存固定为 1/16”的结论。正式比较必须固定基座、数据、训练预算、评估集和 seed，并分别报告模型权重、激活、梯度、优化器和峰值显存。

## 7. 三大创新的关系

```
NF4                 → 基座模型从 16bit 压到 4bit（省 3/4 显存）
Double Quantization → 量化常数的额外开销再砍 75%
Paged Optimizer     → 优化器状态卸到 CPU，消除显存峰值

三者叠加使论文中的 65B/48GB 训练设置成为可能；可迁移性仍取决于实现和训练配置。
```

## 8. 候选起始配置（不是已验证最优值）

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
    r=8,                                  # 候选 rank，需消融
    lora_alpha=16,                        # 候选 alpha，需与 rank/LR 联合检查
    target_modules=["q_proj", "v_proj"],  # 候选模块集合，需按任务比较
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
# 精确可训练参数量应以 model.print_trainable_parameters() 的实际输出为准
```
