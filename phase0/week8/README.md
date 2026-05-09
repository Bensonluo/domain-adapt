# Week 8: 评估方法论 + Phase 0 总结

> 目标: 建立 LLM 评估能力,为后续 Phase 1 实验提供评估基础设施。
> 预计时间: 14-20 小时

> **上周回顾**: Week 7 推导了核心数学 — attention 梯度、softmax+CE、LoRA SVD、DPO、AdamW。这周回到工程: 你在 Week 6 训练的领域模型到底好不好? 怎么量化地回答这个问题?
>
> **为什么学这周**: 不会评估 = 不会改进。评估是你所有实验迭代的起点。你的方向是 domain adaptation — 每次换一个领域数据,都需要快速评估效果。Benchmark + LLM-as-Judge + 人工评估,三层体系缺一不可。
>
> **思考锚点**: "如果你的 SFT 模型在 MMLU 上分数比基座低,但在领域测试上更好,哪个更可信? 为什么?"

---

## Day 1-2: lm-evaluation-harness

> **思考**: 为什么不用 "让模型回答问题然后看对不对" 这种简单评估? MMLU 的 few-shot 评估方法比直接问好在哪里?

### 做什么
1. 安装: `pip install lm-eval`
2. 在 Qwen2.5-3B 基座上跑医疗相关 MMLU 子集
3. 在领域 SFT 模型上跑同样的 tasks
4. 记录: 基座 vs SFT 的分数对比

### 跑
```bash
# 基座模型
python phase0/utils/eval_baseline.py

# SFT 后模型
python phase0/utils/eval_baseline.py \
    --model phase0/week6/domain-sft-merged \
    --output phase0/results/eval_after_sft.json
```

### 怎么解读结果
- MMLU 分数下降 = 领域 SFT 可能损害了通用能力 (catastrophic forgetting)
- MMLU 分数不变/略升 = SFT 没有灾难性遗忘,理想的域适应结果
- 医疗子集分数应该上升 = 领域知识确实增强了

### 交付物
- `phase0/results/baseline_qwen25_3b.json` — 基座分数
- `phase0/results/eval_after_sft.json` — SFT 后分数
- 对比分析

---

## Day 3-4: LLM-as-Judge 范式

> **思考**: LLM judge 自己有 bias — 位置 bias (总是选 A)、长度 bias (总是选更长的)。`judge_with_swap` 怎么减轻位置 bias? 长度 bias 怎么减轻?

### 做什么
1. 理解 pairwise judge 原理
2. 阅读 `llm_as_judge.py`,理解:
   - `parse_winner`: 怎么从 LLM 输出中提取判断结果
   - `judge_with_swap`: A/B 位置互换跑两次,取一致结果
3. 在 20 个领域测试问题上运行
4. 分析结果: A 赢了多少? 位置不一致的有多少?

### 跑
```bash
python phase0/week8/llm_as_judge.py \
    --questions data/processed/domain_test.jsonl \
    --model_a Qwen/Qwen2.5-3B-Instruct \
    --model_b phase0/week6/domain-sft-merged
```

### 交付物
- `phase0/week8/llm_as_judge.py` — judge 实现
- `phase0/results/week8_judge_results.json` — 评估结果

---

## Day 5: 人工评估设计

> **思考**: Cohen's Kappa = 0.3 意味着什么? 两个评分者的一致性够不够?

### 做什么
1. 设计 50-100 个领域评估问题
2. 设计评分 rubric
3. 找 1-2 个人做独立评分
4. 计算 IAA (Cohen's Kappa)

### Kappa 解读
| Kappa | 一致性 |
|-------|--------|
| < 0.2 | 很差 |
| 0.2-0.4 | 一般 |
| 0.4-0.6 | 中等 |
| 0.6-0.8 | 较好 |
| > 0.8 | 很好 |

### 交付物
- `phase0/week8/rubric.md` — 评分标准
- `phase0/results/week8_human_eval.csv` — 评分表
- IAA 计算结果

---

## Day 6-7: Phase 0 总结 + 复盘

### 做什么
1. 回顾 8 周所有交付物,检查哪些已完成
2. 列出已建立的能力和仍有的 gap
3. 整理知识图谱
4. 规划 Phase 1 切入点

### 知识图谱模板
```
Phase 0 能力树:
├── 基础理解
│   ├── Autograd ✓/✗
│   ├── Attention 机制 ✓/✗
│   └── Transformer 架构 ✓/✗
├── 工程能力
│   ├── nanoGPT 训练 ✓/✗
│   ├── HF Trainer 使用 ✓/✗
│   └── 数据处理管道 ✓/✗
├── 进阶方法
│   ├── LoRA/QLoRA ✓/✗
│   ├── SFT + Loss Masking ✓/✗
│   └── 领域模型训练 ✓/✗
└── 评估能力
    ├── Benchmark 评估 ✓/✗
    ├── LLM-as-Judge ✓/✗
    └── 人工评估设计 ✓/✗
```

### 交付物
- `phase0/week8/knowledge_graph.md` — 知识图谱
- `phase0/notes/phase0_summary.md` — Phase 0 总结
- Phase 1 切入点规划

---

## 自测题

1. **Benchmark 评估和 LLM-as-Judge 各自的局限是什么?** 为什么需要人工评估作为补充?
2. **位置 bias 是什么?** `judge_with_swap` 怎么减轻它?
3. **如果你的 SFT 模型在所有 MMLU 子集上都比基座低 2-3 个点,但在领域测试上明显更好,你会继续优化还是接受这个 trade-off?**

> 答案: 1) Benchmark 只能测通用能力,不覆盖你的具体领域;LLM-as-Judge 有自身 bias (位置/长度/自评);人工评估成本高但最可信。三层互补。2) 位置 bias = LLM judge 倾向于选择放在前面的回答。swap 方法: 同一对回答跑两次 (AB 和 BA),如果两次结果一致才采用,不一致则记为 tie。3) 这是典型的 domain adaptation trade-off。通常可以接受 — 2-3 点 MMLU 下降是轻微的通用能力损失,换来了显著的领域能力提升。如果下降 > 5 点,需要考虑数据质量或训练策略。

---

## 验收清单

- [ ] Benchmark 评估报告 (基座 vs SFT)
- [ ] LLM-as-judge 实现代码
- [ ] 人工评估 rubric + 初步评分 + IAA
- [ ] Phase 0 知识图谱
- [ ] Phase 1 切入点规划
- [ ] 自测题能回答 2/3 以上
