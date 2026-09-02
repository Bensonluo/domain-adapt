# Week 8: 评估方法论 + Phase 0 总结

> 目标: 建立 LLM 评估能力,为后续 Phase 1 实验提供评估基础设施。
> 预计时间: 14-20 小时
>
> **本周交付**：已完成结构化评估报告、judge 原型与人工评分模板；开放式评分尚未运行。详见 [CORRECTIONS.md](CORRECTIONS.md)。

> **前后衔接**: Week 7 整理 attention 梯度、softmax+CE、LoRA SVD、DPO 与 AdamW。本周转向评估，量化领域适配的效果与取舍。
>
> **为什么学这周**: 不会评估 = 不会改进。评估方式应与任务匹配：有可靠 ground truth 的结构化任务优先自动评估；开放式任务再组合 benchmark、盲化人工评估和经过校准的 LLM judge。不是所有任务都机械要求三层方法。
>
> **思考锚点**: SFT 模型在 MMLU 上低于基座、在领域测试上更好时，应如何解释两类指标和能力取舍?

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
3. **SFT 模型在所有 MMLU 子集上都比基座低 2–3 个点，但在领域测试上明显更好时，如何判断继续优化还是接受这个 trade-off?**

> 答案: 1) Benchmark 覆盖有限；LLM-as-Judge 有位置、长度和自评偏差；人工评估成本高且也需要 rubric/IAA。应按任务选择互补方法。2) 位置 bias = judge 倾向放在特定位置的回答；swap 只能缓解，不能消除全部偏差。3) 是否接受通用能力下降必须在实验前根据用途定义门槛并报告置信区间，不能事后用固定 2-3/5 点规则决定。

---

## 验收清单

- [x] 独立 `master_data` 案例的结构化评估报告
- [ ] 同一 Week 6 模型的通用 benchmark (基座 vs SFT)
- [x] LLM-as-judge 原型代码
- [ ] LLM-as-judge 运行结果与校准记录
- [x] 人工评估 rubric 模板
- [ ] 盲化人工评分 + 分维度 IAA
- [x] Phase 0 知识图谱
- [x] Phase 1 切入点规划
- [ ] 自测题能回答 2/3 以上
