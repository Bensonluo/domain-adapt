# Phase 0 评估报告

> 数据来源: [4bit-QLoRA-post-training @ 9267c7c](https://github.com/luopeng/4bit-QLoRA-post-training/tree/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/master_data) 项目 master_data 领域评估（2026-05-31）
> 评估脚本: [`domains/master_data/eval/evaluate.py`](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/master_data/eval/evaluate.py)
> 评估集: 800 条（institution 400 + product 400）；这是独立 `master_data/Gemma 26B` 案例，不是 Week 6 `medical_entity/Qwen` 模型的后续评估

> **2026-08-28 方法论纠错**：以下数值保留为历史观测，但旧评估集经过多轮迭代后应视为 dev。base 与 finetuned 的 runtime、temperature、max tokens 不完全一致，因此差异代表两条完整推理链路之差，不能全部归因于 LoRA/SFT。数据来自同源合成生成流程，真实业务外部效度尚未验证。详见 [CORRECTIONS.md](CORRECTIONS.md)。

---

## 1. 任务内结构化评估 — Base vs Finetuned

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
- 机构匹配观测到 +19 个百分点，说明当前微调链路在该同源合成分布上有强提升信号；公平同链路重跑前不能隔离 SFT 的纯因果贡献
- 解析失败从 10 降到 0：微调后模型输出格式更稳定
- 延迟观测不可直接比较：finetuned 使用本地 MLX，base 使用 LM Studio/API，且解码配置不同
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

**适用范围**：该表保留为描述性历史记录。原始运行混有不同样本量、runtime、prompt/解码链路，因此不能作为统一 benchmark，也不能证明稳定超越所有更大模型或商业 API。正式排行榜必须让所有模型运行同一完整 blind test，并固定 prompt、输出 schema 和解码参数；延迟需在相同硬件/服务条件下单独比较。

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
- **数据分布 bias**: 已执行部分编码/query 层隔离，但仍需报告目标实体 code、normalized query、模板和候选集合四层 overlap；同一生成器产生的 train/test 不能概括为“零泄漏”

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
2. **LoRA rank=32**: 当前配置选择了 rank=32；没有匹配条件的 r=8/16/32 消融，不能声称 rank=32 必要
3. **7 个 target modules**: 当前配置覆盖注意力层 + FFN 层；是否优于只训 q/v 仍需参数预算匹配的消融
4. **数据构造**: 6000 条数据包含硬负采样与噪声注入；没有 50K 低质量对照，不能得出“远优于 50K”的因果结论

---

## 5. 综合结论

### 当前可以支持的结论

- 在当时两条完整推理链路和同源合成评估上，机构匹配 Top-1 从 79.75% → 98.75%（+19 个百分点）
- 药品匹配 B 级精度从 93.5% → 99.75%（+6.3%）
- 模型输出格式稳定性提升（parse failure 10 → 0）
- 两条链路的延迟观测不同，但因 runtime 与解码混杂，不归因于微调

该结果支持“结构化合成匹配 POC 有强提升信号”，尚不支持真实业务外部效度、LoRA 单独因果贡献或统一跨模型领先。

### 意外发现

1. **小数据强信号**: 6000 条构造数据在该 POC 分布上取得了明显提升；跨模型结论待统一协议验证
2. **rank=32 是当前选择**: 必要性和 r=8 是否不足尚未验证
3. **输出更稳定**: parse failure 下降是可复核观测；延迟原因需独立系统 benchmark 分解

### 下一步（Phase 1 方向）

1. **DPO 实战**: 用偏好数据（正确匹配 vs 错误匹配）做 DPO 训练，进一步提升置信度校准
2. **多领域扩展**: 从主数据匹配扩展到更多业务场景（如药品相互作用、处方审核）
3. **生产部署**: MLX → vLLM / TGI 部署，量化延迟和吞吐量
4. **评估体系升级**: 引入 LLM-as-Judge 评估开放域问答，不只是结构化匹配
