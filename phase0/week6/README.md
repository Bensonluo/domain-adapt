# Week 6: 第一次完整领域模型训练

> 目标: 把 Week 1-5 的所有知识整合,训练一个完整的领域 SFT 模型。
> 预计时间: 14-20 小时
>
> **本周交付**：实际交付为外部药品实体匹配 SFT 案例，原先设想的开放式任务未开展。详见 [CORRECTIONS.md](CORRECTIONS.md)。

> **前后衔接**: 将 Week 5 的 chat template、loss masking 和数据质量检查组合到端到端领域 SFT 流程中。
>
> **实践重点**: 串联数据准备、训练、模型保存和初步评估，记录各环节的选择依据与问题定位过程。
>
> **分析重点**: 从数据清洗到模型评估，各阶段均可能影响最终结果。通过分阶段记录与检查，定位效果变化对应的环节。
>
> 定位思路：先检查样本与标签，再检查模板及 loss mask，随后对照训练日志和逐题输出。这样可以区分数据问题、训练目标变化与推理配置差异。

---

## Day 1-3: 准备领域 SFT 数据

> **思考**: 为什么数据去重 (deduplication) 这么重要? 训练集和测试集有重复会怎样? (提示: 测试集污染 → 虚高的评估分数)

### 做什么
1. 合成领域对话数据,或用公开数据集
2. 数据格式化: 统一成 JSONL 格式
3. 数据清洗: MD5 精确去重、最低长度启发式、格式校验；长度规则不能替代正确性/来源/覆盖/难度和人工抽检
4. 构建 2000-5000 条高质量领域指令数据
5. 按 prompt、实体、来源或生成模板分组划分 train/dev/test；随机行切分必须显式说明不存在组级依赖

### 公开数据集
- HuatuoGPT: https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1
- ChatMed: https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset

### 跑
```bash
python phase0/week6/dataset_prep.py \
    --input raw_data.jsonl \
    --output domain_sft.jsonl \
    --eval_output domain_dev.jsonl \
    --group_field entity_id \
    --seed 42
```

### 交付物
- `phase0/data/processed/domain_sft.jsonl`
- `phase0/data/processed/domain_dev.jsonl`
- 独立来源、冻结后只运行一次的 `domain_blind_test.jsonl`（不能由本脚本同池切分后反复调参）
- 数据清洗报告（精确去重数、长度过滤数、错误类型；另附独立质量审计）

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
- [ ] Loss masking 生效：每条样本有监督 token、prompt 被 mask、assistant 结束 token被监督；记录比例但不设通用 50% 阈值
- [ ] target_modules、可训练参数量和选择理由已记录；是否扩展模块由消融决定

训练中观察:
- [ ] loss、梯度范数和 dev 指标按预定间隔记录；不预设所有任务前 100 步必须快速下降
- [ ] loss 不降时联合检查数据目标、有效 batch、learning rate、冻结参数和 truncation
- [ ] loss 为 NaN 时检查数值精度、学习率、异常样本、梯度和 masking，不把原因限定为 masking

### 跑 (GPU 服务器)
```bash
cd /root/workspace/domain-adapt/phase0/week6
python domain_sft.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data ../../data/processed/domain_sft.jsonl \
    --eval_data ../../data/processed/domain_dev.jsonl \
    --output_dir ./domain-sft

# 合并 adapter
python merge_adapter.py --adapter ./domain-sft --output ./domain-sft-merged
```

`--data` 与 `--eval_data` 必须在训练前按实体、prompt 或来源分组切分。脚本每个 epoch 计算 dev loss，并恢复 `eval_loss` 最低的 checkpoint；最终 blind test 不得用于该选择过程。

### 交付物
- Adapter 权重 (`domain-sft/`)
- 合并后的完整模型 (`domain-sft-merged/`)
- 训练 loss 曲线

---

## Day 7: 初步评估

> **思考**: 人工评估 20 题能说明什么? 不能说明什么? (为什么 Week 8 需要更系统的方法)

### 做什么
1. 20 题仅作 smoke test，不作为正式效果结论；题目覆盖不同场景和难度
2. base 与 finetuned 使用同 runtime、模板、解码参数，盲化模型身份和回答顺序
3. 保存逐题评分、rubric 和错误切片；若扩大评估，可增加题目覆盖与独立评分者

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

> 答案: 1) QLoRA 的量化基座权重较省显存，主要开销通常来自激活、LoRA 梯度/优化器状态和临时 buffer，具体占比依赖序列长度与实现。2) 必须 fail-closed 并报错；静默退化为全序列 loss 会改变训练目标、污染实验。3) merge 是把 ΔW 加回 W_0，数学上等价，但量化、反量化和浮点精度可能产生差异，应做数值/生成一致性检查。

---

## 验收清单

- [ ] 2000-5000 条领域数据集 (train/test split)
- [ ] 完整领域 SFT 模型 (训练脚本 + 配置文件)
- [ ] 人工评估记录 (20 题 × 2 模型对比)
- [ ] 训练 loss 曲线截图
- [ ] 自测题能回答 2/3 以上

---

## 成果

Week 6 的实践交付来自 [4bit-QLoRA-post-training](https://github.com/luopeng/4bit-QLoRA-post-training/tree/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126)，证据固定到 commit `9267c7c569eeb9f2b14d0a1cf0faa67c831d7126`。已完成结构化实体匹配 SFT 案例；模型、数据、任务和评估与原计划不完全一致，人工评估交付物仍缺失。

**数据准备** — 从 14K+ 药品知识库生成 58K+ Alpaca 格式训练样本，含硬负采样、噪声注入，并按药品编码分组划分 train/val/test。该规则降低了实体编码层面的直接重叠，但不等于已经排除模板、归一化 query、候选集合和同生成器分布重合。见 [prepare_data.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/prepare_data.py) 和 [train.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/data/train/train.json)。

**QLoRA 训练** — 支持 7 个预设（mac/poc/full 等），覆盖 Qwen3-1.7B 到 Qwen3-14B，含 MLflow + TensorBoard 追踪。见 [train_medical_entity.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/scripts/train_medical_entity.py) 和 [训练配置](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/config/domains/medical_entity.py)。

**评估** — 分难度 accuracy 对比（base vs finetuned），10+ 次评估迭代。该 commit 中记录的最新结果为 hard 难度 66.7%。见 [evaluate.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/evaluate.py) 和 [评估结果](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/data/results/executive_summary_20260528_193258.md)。

### 详细说明

#### 数据工程

[prepare_data.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/prepare_data.py) 从 14K+ 药品知识库 (`drug_knowledge_base.json`) 动态生成训练样本。每条样本包含一个查询实体和多个候选实体，模型需要从候选中选出正确的标准名称。

负采样策略分三层：
1. **同通用名不同剂型**（最强硬负例）— 如"恩替卡韦片" vs "恩替卡韦胶囊"
2. **名称前缀相似** — 如"阿魏酸钠注射液" vs "阿魏酸钠片"
3. **随机负例** — 补充多样性

数据增强包括噪声注入（随机替换/删除/插入字符），模拟真实场景中的错别字。按药品编码划分 train/val/test，用于控制目标实体编码层面的直接重叠；进一步分析可检查 normalized query、模板和候选集合 overlap，不能把单层分组笼统称为“零泄漏”。

最终产出 [train.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/data/train/train.json)（58K+ 条）、[val.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/data/val/val.json)（7K+ 条）、[test_instruction.json](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/data/test/test_instruction.json)（7K+ 条），远超 Week 6 要求的 2000-5000 条。

#### QLoRA 训练

[训练配置](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/config/domains/medical_entity.py) 提供 7 个预设，适配不同硬件和模型规模：

| 预设 | 模型 | LoRA r/alpha | 硬件 |
|---|---|---|---|
| mac | Qwen3-14B | 64/128 | Mac 64GB |
| mac-8b | Qwen3-8B | 32/64 | Mac 64GB |
| mac-small | Qwen3-4B | 32/64 | Mac 64GB |
| mac-35-4b | Qwen3.5-4B | 32/64 | Mac 64GB |
| mac-2b | Qwen3.5-2B | 16/32 | Mac 64GB |
| mac-1b | Qwen3-1.7B | 16/32 | Mac 64GB |
| poc | Qwen3-4B 4bit | 32/64 | 8GB GPU |

所有预设的 target_modules 包含 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`，覆盖注意力层和 FFN 层。训练使用 cosine LR scheduler + 5% warmup，集成 [MLflow](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/src/tracking/mlflow_tracker.py) 和 TensorBoard 追踪。Adapter 合并通过 [merge_lora.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/scripts/merge_lora.py) 完成。

#### 评估

[evaluate.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/evaluate.py) 对测试集逐条推理，与启发式基线对比。评估报告由 [report.py](https://github.com/luopeng/4bit-QLoRA-post-training/blob/9267c7c569eeb9f2b14d0a1cf0faa67c831d7126/domains/medical_entity/eval/report.py) 自动生成，包含分难度 accuracy、成本估算（延迟/吞吐量/日处理量）。

最新评估（2026-05-28）：finetuned model hard 难度 accuracy 66.7%，与启发式基线 67.3% 接近。当前瓶颈在困难样本（错别字/口语化表述），后续可通过增强噪声注入、增加训练轮次或升级基座模型来提升。
