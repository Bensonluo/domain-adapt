# Week 6: 第一次完整领域模型训练

> 目标: 把 Week 1-5 的所有知识整合,训练一个完整的领域 SFT 模型。
> 预计时间: 14-20 小时

> **上周回顾**: Week 5 你掌握了 SFT 的每个组件 — chat template、loss masking、数据质量。这周是 Phase 0 的实战高潮: 整合所有知识,端到端地训练一个你自己的领域模型。
>
> **为什么学这周**: 这是 Phase 0 的最终交付。从这周开始,你不再是在做练习 — 你在做真实的 domain adaptation。训练出来的模型会在 Week 8 被严格评估,让你看到 "数据+算法+工程" 的组合效果。
>
> **思考锚点**: "从数据清洗到模型评估,每一步都可能有 bug。我怎么确保最终效果不好时,能定位到是哪一步出了问题?"

---

## Day 1-3: 准备领域 SFT 数据

> **思考**: 为什么数据去重 (deduplication) 这么重要? 训练集和测试集有重复会怎样? (提示: 测试集污染 → 虚高的评估分数)

### 做什么
1. 合成领域对话数据,或用公开数据集
2. 数据格式化: 统一成 JSONL 格式
3. 数据清洗: 去重(MD5)、过滤低质量(长度<10字)、格式校验
4. 构建 2000-5000 条高质量领域指令数据
5. 划分 train/test (90/10)

### 公开数据集
- HuatuoGPT: https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1
- ChatMed: https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset

### 跑
```bash
python phase0/week6/dataset_prep.py --input raw_data.jsonl --output domain_sft.jsonl
```

### 交付物
- `phase0/data/processed/domain_sft.jsonl`
- `phase0/data/processed/domain_test.jsonl`
- 数据清洗报告 (去重数、过滤数、质量分布)

---

## Day 4-6: 完整训练流程

> **思考**: 为什么要 merge adapter 回 base model? 不 merge 直接用行不行? (提示: 可以,但推理时多一次 LoRA 计算,而且部署环境需要 peft 库)

### 做什么
1. 加载 Qwen2.5-3B-Instruct
2. 配置 QLoRA (r=16, alpha=32)
3. 训练,监控 loss 曲线
4. 保存 adapter → 合并到 base model

### 关键检查点
训练开始前确认:
- [ ] BitsAndBytesConfig 使用 NF4 + double_quant
- [ ] `prepare_model_for_kbit_training` 已调用
- [ ] Loss masking 生效 (检查 labels 中 -100 的比例 > 50%)
- [ ] target_modules 包含 q/k/v/o_proj

训练中观察:
- [ ] 前 100 步 loss 应该快速下降
- [ ] 如果 loss 不降 → 检查 learning rate
- [ ] 如果 loss 为 NaN → 检查 loss masking 是否正确

### 跑 (GPU 服务器)
```bash
cd /root/workspace/domain-adapt/phase0/week6
python domain_sft.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data ../../data/processed/domain_sft.jsonl \
    --output_dir ./domain-sft

# 合并 adapter
python merge_adapter.py --adapter ./domain-sft --output ./domain-sft-merged
```

### 交付物
- Adapter 权重 (`domain-sft/`)
- 合并后的完整模型 (`domain-sft-merged/`)
- 训练 loss 曲线

---

## Day 7: 初步评估

> **思考**: 人工评估 20 题能说明什么? 不能说明什么? (为什么 Week 8 需要更系统的方法)

### 做什么
1. 人工测试 20 个领域问题(覆盖不同场景和难度)
2. 对比: 原始 Qwen2.5-3B vs 微调后的回答
3. 记录评分(1-5)和幻觉频率

### 跑
```bash
python phase0/week6/eval_manual.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --finetuned_model ./domain-sft-merged \
    --questions ../../data/processed/domain_test.jsonl
```

### 交付物
- `phase0/results/week6_manual_eval.md` — 人工评估记录
- 20 题 × 2 模型对比评分表

---

## 自测题

1. **QLoRA 训练时显存主要花在哪里?** 模型权重? 梯度? 优化器状态? 激活值?
2. **domain_sft.py 中 `mask_assistant_labels` 如果找不到 assistant marker 会怎样?** 这是好的 fallback 吗?
3. **merge adapter 后的模型和 merge 前的推理结果是否完全一致?** (数学上,不考虑数值精度)

> 答案: 1) QLoRA 的模型权重只有 4bit,很省。主要花在: 优化器状态 (LoRA 参数的 AdamW m/v,虽然少但是 FP32) + 激活值 (gradient checkpointing 可以缓解)。2) 会 fallback 到全部计入 loss (不做 masking)。不算好 — 意味着会训练 prompt 部分。但总比报错好。3) 是的,merge 就是把 ΔW 加回 W_0,数学上等价,但浮点精度可能有微小差异。

---

## 验收清单

- [ ] 2000-5000 条领域数据集 (train/test split)
- [ ] 完整领域 SFT 模型 (训练脚本 + 配置文件)
- [ ] 人工评估记录 (20 题 × 2 模型对比)
- [ ] 训练 loss 曲线截图
- [ ] 自测题能回答 2/3 以上

---

## 成果

本项目所有 Week 6 交付物已在实战项目 [4bit-QLoRA-post-training](https://github.com/luopeng/4bit-QLoRA-post-training) 中完成。

**数据准备** — 从 14K+ 药品知识库生成 58K+ Alpaca 格式训练样本，含硬负采样、噪声注入、按药品编码零泄漏划分 train/val/test。见 [prepare_data.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/prepare_data.py) 和 [train.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/data/train/train.json)。

**QLoRA 训练** — 支持 7 个预设（mac/poc/full 等），覆盖 Qwen3-1.7B 到 Qwen3-14B，含 MLflow + TensorBoard 追踪。见 [train_medical_entity.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/scripts/train_medical_entity.py) 和 [训练配置](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/config/domains/medical_entity.py)。

**评估** — 分难度 accuracy 对比（base vs finetuned），10+ 次评估迭代。最新结果：hard 难度 66.7%。见 [evaluate.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/evaluate.py) 和 [评估结果](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/data/results/executive_summary_20260528_193258.md)。

### 详细说明

#### 数据工程

[prepare_data.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/prepare_data.py) 从 14K+ 药品知识库 (`drug_knowledge_base.json`) 动态生成训练样本。每条样本包含一个查询实体和多个候选实体，模型需要从候选中选出正确的标准名称。

负采样策略分三层：
1. **同通用名不同剂型**（最强硬负例）— 如"恩替卡韦片" vs "恩替卡韦胶囊"
2. **名称前缀相似** — 如"阿魏酸钠注射液" vs "阿魏酸钠片"
3. **随机负例** — 补充多样性

数据增强包括噪声注入（随机替换/删除/插入字符），模拟真实场景中的错别字。按药品编码划分 train/val/test，确保训练集和测试集的药品完全不重叠，避免数据泄漏。

最终产出 [train.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/data/train/train.json)（58K+ 条）、[val.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/data/val/val.json)（7K+ 条）、[test_instruction.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/data/test/test_instruction.json)（7K+ 条），远超 Week 6 要求的 2000-5000 条。

#### QLoRA 训练

[训练配置](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/config/domains/medical_entity.py) 提供 7 个预设，适配不同硬件和模型规模：

| 预设 | 模型 | LoRA r/alpha | 硬件 |
|---|---|---|---|
| mac | Qwen3-14B | 64/128 | Mac 64GB |
| mac-8b | Qwen3-8B | 32/64 | Mac 64GB |
| mac-small | Qwen3-4B | 32/64 | Mac 64GB |
| mac-35-4b | Qwen3.5-4B | 32/64 | Mac 64GB |
| mac-2b | Qwen3.5-2B | 16/32 | Mac 64GB |
| mac-1b | Qwen3-1.7B | 16/32 | Mac 64GB |
| poc | Qwen3-4B 4bit | 32/64 | 8GB GPU |

所有预设的 target_modules 包含 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`，覆盖注意力层和 FFN 层。训练使用 cosine LR scheduler + 5% warmup，集成 [MLflow](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/src/tracking/mlflow_tracker.py) 和 TensorBoard 追踪。Adapter 合并通过 [merge_lora.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/scripts/merge_lora.py) 完成。

#### 评估

[evaluate.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/evaluate.py) 对测试集逐条推理，与启发式基线对比。评估报告由 [report.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/main/domains/medical_entity/eval/report.py) 自动生成，包含分难度 accuracy、成本估算（延迟/吞吐量/日处理量）。

最新评估（2026-05-28）：finetuned model hard 难度 accuracy 66.7%，与启发式基线 67.3% 接近。当前瓶颈在困难样本（错别字/口语化表述），后续可通过增强噪声注入、增加训练轮次或升级基座模型来提升。
