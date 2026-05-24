# PEFT LoRA 源码核心要点

源码位置: `peft/tuners/lora/layer.py` (~2500 行)，核心类 `Linear(nn.Module, LoraLayer)`

## 1. Forward — 和手写版的 3 个关键差异

```python
# PEFT 官方 forward (line 941-982)
def forward(self, x):
    result = self.base_layer(x)                    # 原始层输出
    for active_adapter in self.active_adapters:     # 支持多适配器
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = self._cast_input_dtype(x, lora_A.weight.dtype)  # 类型对齐
        result = result + lora_B(lora_A(dropout(x))) * scaling
    return result
```

```python
# 手写版 forward
def forward(self, x):
    original = self.original_linear(x)
    lora_update = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    return original + lora_update
```

**差异 1: 用 nn.Linear 而不是裸参数**
PEFT 把 lora_A/lora_B 包装成 `nn.Linear`，不是裸的 `nn.Parameter`。好处：自动处理 bias、device、dtype 转换。

**差异 2: 支持 Dropout**
PEFT 在 A 之前加了 `lora_dropout(x)`，防止小数据集过拟合。手写版没有。

**差异 3: 类型安全**
PEFT 有 `_cast_input_dtype` 确保 x 和 lora_A 的 dtype 一致（QLoRA 场景下基座是 NF4，LoRA 是 BF16，dtype 不同会报错）。手写版没处理。

## 2. Merge — 推理零开销的秘密

```python
# merge (line 817-883): 把 LoRA 权重合入基座权重
def merge(self, safe_merge=False):
    if safe_merge:
        # 安全模式: 先 clone，检查 NaN，再赋值
        orig_weight = base_layer.weight.data.clone()
        orig_weight += self.get_delta_weight(active_adapter)
        if not torch.isfinite(orig_weight).all():
            raise ValueError("NaNs detected!")
        base_layer.weight.data = orig_weight
    else:
        # 快速模式: 直接原地加
        base_layer.weight.data += self.get_delta_weight(active_adapter)

# unmerge (line 884-905): 从基座权重中减掉 LoRA，恢复原状
def unmerge(self):
    weight.data -= self.get_delta_weight(active_adapter)
```

```python
# get_delta_weight (line 907-939): 计算 ΔW = (B @ A) * scaling
def get_delta_weight(self, adapter):
    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
    return output_tensor
```

**核心流程**: 训练完 → `model.merge_and_unload()` → 基座权重 += ΔW → 丢弃 LoRA 层 → 推理零开销

手写版没实现 merge，要手动 `W.data += (alpha/r) * B @ A`。PEFT 的 `safe_merge=True` 还会检查 NaN，防止坏的适配器污染权重。

## 3. Scaling — rsLoRA 变体

```python
# 普通 LoRA (line 215)
self.scaling[adapter_name] = lora_alpha / r

# rsLoRA (rank-stabilized, line 213)
self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
```

rsLoRA 论文发现：用 `sqrt(r)` 而不是 `r` 做分母，训练更稳定。因为 ΔW 的方差随 rank 变化，`1/sqrt(r)` 能让不同 rank 的更新量级保持一致。

手写版只用 `alpha / r`，这是标准做法，够用。

## 4. 初始化 — 3 种方式

```python
# line 263-276
if init_lora_weights is True:
    # 默认: A 用 Kaiming Uniform，B 全零
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)

elif init_lora_weights == "gaussian":
    # 高斯: A 用正态分布 (std=1/r)，B 全零
    nn.init.normal_(self.lora_A.weight, std=1 / self.r)
    nn.init.zeros_(self.lora_B.weight)

elif init_lora_weights == "pissa":
    # PiSSA: 用 SVD 初始化，A/B 直接取权重的主成分
    V, S, Uh = torch.linalg.svd(weight)
    lora_A = diag(sqrt(S/r)) @ Uh[:r]
    lora_B = V[:, :r] @ diag(sqrt(S/r))
    weight -= scaling * lora_B @ lora_A  # 从基座中减掉
```

手写版: `A = randn * 0.01, B = zeros`，对应 `init_lora_weights="gaussian"` 的简化版。

**PiSSA 的巧妙之处**: 初始化时就把权重的主成分分给 LoRA，基座只保留残差。训练从第一步就有效，不需要等 LoRA 慢慢学到主成分。

## 5. Conv1D 兼容 — fan_in_fan_out

```python
# nn.Linear: weight shape = [out, in]     → B @ A 直接就是 [out, in]
# Conv1D:    weight shape = [in, out]      → B @ A 得到 [out, in]，需要 transpose
# 所以 get_delta_weight 里有:
output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out)
```

GPT-2 用 Conv1D，weight shape 和 nn.Linear 相反。PEFT 通过 `fan_in_fan_out` 标志自动处理转置。手写版在 `__init__` 里根据 `isinstance(original_linear, Conv1D)` 分别取 shape，forward 里用 `x @ A.T @ B.T` 统一处理。

## 6. 多适配器支持

PEFT 用 `ModuleDict` 存储 A 和 B（key 是适配器名字），forward 遍历所有 `active_adapters` 累加增量。手写版只支持单个适配器。

实际用途：可以在同一个模型上加载多个任务的 LoRA，推理时切换 `active_adapters`。

## 7. 必须掌握的知识点总结

| 知识点 | 重要程度 | 说明 |
|--------|---------|------|
| B=0 初始化 | 核心 | 保证训练开始时输出不变 |
| merge/unmerge | 核心 | 推理零开销的关键操作 |
| scaling = alpha/r | 核心 | 控制更新量级，换 rank 不用调学习率 |
| Dropout | 实用 | 防过拟合，小数据集必备 |
| dtype 转换 | QLoRA 必备 | 基座 4bit + LoRA 16bit 需要类型对齐 |
| rsLoRA (1/sqrt(r)) | 进阶 | rank 变化时训练更稳定 |
| PiSSA 初始化 | 进阶 | SVD 初始化加速收敛 |
| fan_in_fan_out | GPT-2 必备 | Conv1D weight shape 转置处理 |
