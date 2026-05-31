# Phase 0 评估报告

> 数据来源: [4bit-QLoRA-post-training](https://github.com/luopeng/4bit-QLoRA-post-training) 项目 master_data 领域评估（2026-05-31）
> 评估脚本: `domains/master_data/eval/evaluate.py`
> 测试集: 800 条（institution 400 + product 400），与训练集零药品编码重叠

---

## 1. Benchmark 评估 — Base vs Finetuned

### 任务说明

医药主数据匹配：给定查询实体 + N 个候选实体，模型需选出正确的标准实体。
- **Institution（机构匹配）**: 查询 "银川市迎宾路社区服务中心" → 从候选列表中找到匹配的标准机构
- **Product（药品匹配）**: 查询 "恩替卡韦片" → 匹配标准药品名 + 评级（A/B/D）

### 基座模型: gemma-4-26b-a4b（未微调）

| Task | Samples | Top-1 Acc | Precision | Recall | F1 | Latency |
|------|---------|-----------|-----------|--------|----|---------|
| Institution | 400 | 79.75% | 85.29% | 79.75% | 82.43% | 21.8s |
| Product | 400 | 100.00% | — | — | — | 21.9s |

### Finetuned 模型: gemma-4-26b-a4b + LoRA adapter

| Task | Samples | Top-1 Acc | Precision | Recall | F1 | Latency |
|------|---------|-----------|-----------|--------|----|---------|
| Institution | 400 | **98.75%** | **99.25%** | **98.75%** | **99.00%** | 7.8s |
| Product | 400 | 100.00% | — | — | — | 11.8s |

Product 额外指标: Grade Accuracy 98.6% → 99.7%（B 级 93.5% → 99.75%）

### 对比分析

| 指标 | Base | Finetuned | Δ |
|------|------|-----------|---|
| Institution Top-1 | 79.75% | **98.75%** | **+19.0%** |
| Institution F1 | 82.43% | **99.00%** | **+16.6%** |
| Product B-grade | 93.50% | **99.75%** | **+6.3%** |
| Parse Failures | 10 | 0 | **-10** |
| Avg Latency | 21.8s | 7.8s | **-64%** |

**关键发现**：
- 机构匹配提升最显著（+19%），验证了 SFT 在领域任务上的有效性
- 解析失败从 10 降到 0：微调后模型输出格式更稳定
- 推理延迟降低 64%：本地 MLX 推理 vs LM Studio API
- Product Top-1 已达天花板（100%），但 B 级精度仍有提升空间

---

## 2. 跨模型排行榜

### Institution Top-1 Accuracy（400 样本）

| Rank | Model | Size | Top-1 Acc | 备注 |
|------|-------|------|-----------|------|
| 🥇 | **gemma-4-26b finetuned** | 26B | **98.75%** | 本地 MLX + LoRA |
| 2 | gemma-4-31b | 31B | 88.0% | LM Studio baseline |
| 3 | GLM-5.1 | — | 85.2% | 智谱云端 API |
| 4 | MiniMax-M2.7 | — | 83.8% | MiniMax 云端 API |
| 5 | gemma-4-26b baseline | 26B | 79.38% | LM Studio baseline |
| 6 | qwen3.6-35b | 35B | 76.0% | LM Studio baseline |
| 7 | qwen3.5-9b | 9B | 72.60% | LM Studio baseline |
| 8 | qwen3-30b | 30B | 72.0% | LM Studio baseline |
| 9 | qwen3-8b | 8B | 62.0% | LM Studio baseline |

**分析**：
- 26B finetuned 模型超越所有更大模型（31B、35B）和商业云端 API
- 证明 domain SFT 的 ROI 远大于单纯增大模型参数
- 未微调的 26B baseline 排名第 5（79.38%），微调后直接跳到第 1（98.75%）
- 最小的 qwen3-8b 仅 62%，说明任务本身有难度，不是所有模型都能做好

---

## 3. LLM-as-Judge 方法论

### 评估方法

本项目采用 **结构化输出评估**（而非自由文本 judge）：
- 模型输出标准 JSON 数组，每个候选一个 `{matched: bool, confidence: string}` 对象
- 评估脚本自动解析 JSON，与 ground truth 逐条比对
- 主指标: Top-1 Selection Accuracy（N 个候选中是否选对了）

### 为什么不用 LLM-as-Judge（自由文本评判）

| 方法 | 优势 | 劣势 |
|------|------|------|
| 结构化输出 + 自动比对 | 客观、可复现、零 bias | 只能评估结构化任务 |
| LLM-as-Judge | 适用于开放性问答 | 有位置 bias、长度 bias、自评 bias |
| 人工评估 | 最可信 | 成本高、不可扩展 |

本项目的匹配任务是结构化任务（选择题），结构化评估比 LLM-as-Judge 更合适。
LLM-as-Judge 适用于 Phase 1 的开放域问答评估（如医疗咨询、用药建议）。

### Bias 分析（参考 Week 8 理论）

即使使用结构化评估，仍需关注：
- **解析 bias**: baseline 模型有 10 次 parse failure，finetuned 为 0 — 输出稳定性也是质量指标
- **模型选择 bias**: 不同基座模型表现差异大（62% ~ 88%），说明模型选择本身影响巨大
- **数据分布 bias**: 测试集按药品编码零泄漏划分，避免训练集污染

---

## 4. 训练配置回顾

### 最终采用的配置

```yaml
model: gemma-4-26b-a4b (4-bit MLX)
LoRA:
  rank: 32
  scale: 64.0
  keys: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  dropout: 0.05
训练:
  iters: 10000
  batch_size: 1
  learning_rate: 1e-4
  lr_schedule: cosine_decay (warmup 500)
  max_seq_length: 1280
  mask_prompt: true  ← Loss Masking ✅
数据:
  train: 6000 条 (3000 institution + 3000 product)
  test: 800 条 (400 institution + 400 product)
  零药品编码泄漏划分
```

### 关键决策

1. **Loss Masking**: `mask_prompt: true`，只在 assistant 的 JSON 匹配结果上计算 loss
2. **LoRA rank=32**: 比 QLoRA 论文推荐的 r=8 大，因为匹配任务需要学习复杂的推理规则
3. **7 个 target modules**: 覆盖注意力层 + FFN 层，比只训 q/v_proj 效果好
4. **数据质量**: 6000 条精心构造的数据（含硬负采样 + 噪声注入）远优于 50K 条低质量数据

---

## 5. 综合结论

### Domain Adaptation 是否成功？ ✅ 是

- 机构匹配 Top-1 从 79.75% → 98.75%（+19%），超越所有商业 API
- 药品匹配 B 级精度从 93.5% → 99.75%（+6.3%）
- 模型输出格式稳定性提升（parse failure 10 → 0）
- 本地推理延迟降低 64%（去掉 API 开销）

### 意外发现

1. **小数据大效果**: 6000 条数据足以让 26B 模型在特定任务上超越 35B 模型
2. **rank=32 必要**: 匹配任务需要较大的 LoRA rank，r=8 不够（早期实验验证）
3. **延迟反降**: 微调后模型输出更简洁（直接出 JSON），减少了解码步数

### 下一步（Phase 1 方向）

1. **DPO 实战**: 用偏好数据（正确匹配 vs 错误匹配）做 DPO 训练，进一步提升置信度校准
2. **多领域扩展**: 从主数据匹配扩展到更多业务场景（如药品相互作用、处方审核）
3. **生产部署**: MLX → vLLM / TGI 部署，量化延迟和吞吐量
4. **评估体系升级**: 引入 LLM-as-Judge 评估开放域问答，不只是结构化匹配
