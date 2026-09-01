# Phase 1：深钻阶段 — Deep Dive

> Peng Luo 1-2 年规划第二步 | 13 周 + 数据准备 | 主攻深度：蒸馏 + GRPO
>
> Phase 0 完成了主要学习产物，但研究性验收仍为 `REMEDIATION_REQUIRED`。
> Phase 1 在此基础上深钻 4 个方向，补齐短板。

## 当前状态（唯一阶段入口）

`REMEDIATION_REQUIRED`

Week 9–21 的学习和实验执行大部分完成；grouped preference split、评估登记、五类确认规范和可恢复历史重分析已补齐。当前阻断项收敛为：外部 blind evaluation 尚不可用，以及冻结的多 seed 受控确认实验尚未执行。既有 CMExam 500 题与反复使用的 CMMLU 子集统一视为 **development benchmark**，不能再作为 Phase 1 最终 blind test。

- 声明—证据矩阵：[`audit/claim-evidence-matrix.md`](audit/claim-evidence-matrix.md)
- 退出门槛：[`audit/exit-gate.md`](audit/exit-gate.md)
- 最小确认实验：[`audit/rerun-plan.md`](audit/rerun-plan.md)
- 证据清单：[`artifact_manifest.md`](artifact_manifest.md)
- 机器可读证据注册表：[`artifact_registry.json`](artifact_registry.json)
- 评估隔离与盲测协议：[`audit/blind-test-protocol.md`](audit/blind-test-protocol.md)
- Benchmark 污染登记：[`audit/benchmark_registry.json`](audit/benchmark_registry.json)
- 治理决策：[`audit/ADR-001-evidence-registry-and-blind-test.md`](audit/ADR-001-evidence-registry-and-blind-test.md)
- 审计运行入口：[`audit/README.md`](audit/README.md)
- 五类确认实验冻结规范：[`confirmation/README.md`](confirmation/README.md)

| 周 | 执行动作 | 证据 | 当前方法学判定 |
|---|---|---|---|
| W8.5 | PARTIAL | PARTIAL | 原数据计划未完整实现；实际由 W10/W14 的替代路线承接 |
| W9 | DONE | PRESENT | 理论笔记完成；部分普适表述需收窄 |
| W10 | DONE | PRESENT | 管线完成；1B–3B token 规模目标未完成 |
| W11 | DONE | PRESENT | demo CPT 跑通，不构成严肃 CPT 证据 |
| W12 | DONE | PRESENT | 多配比已跑；单 seed、测试复用和因果混杂待修 |
| W13 | DONE | PRESENT | 理论推导完成 |
| W14 | DONE | PARTIAL | 新 grouped split 已实现且 prompt overlap=0；历史 split 有 50 组交叉，独立质量验证待补 |
| W15 | DONE | PRESENT | 历史 beta sweep 使用泄漏 split；“最优 beta”撤回，确认实验必须使用 grouped v1 |
| W16 | DONE | PRESENT | 失败模式实验完成；IPO 信号待独立确认 |
| W17 | DONE | INVALIDATED_PART | 8/500 题训练重叠；clean delta 仅可界定为 +0.61pp～+3.86pp，缺 450 条逐题预测，确认性迁移仍失效 |
| W18 | DONE | PRESENT | 理论笔记完成；on-policy 普适优越性需收窄 |
| W19 | DONE | PRESENT | 三臂完成；机制解释仍是待验证假设 |
| W20 | DONE | REPRODUCIBLE_PART | lineage 较完整；多臂单 seed 的强结论待确认 |
| W21 | DONE | REPRODUCIBLE_PART | clean matched 确认数据已就绪；多 seed 非劣效重训、外部 blind 和临床人工审核未闭环 |

---

## 总览

系统补齐 4 个 P0 短板：
1. **Continual Pre-training (CPT)** — 让模型真正理解领域
2. **DPO / GRPO / Preference Tuning** — 对齐模型偏好
3. **Knowledge Distillation** — 把大模型能力迁移到小模型
4. **Synthetic Data Generation** — 用数据工程放大训练效果

**主攻深度方向**：蒸馏 + GRPO

---

## 目录结构

```
phase1/
├── README.md              ← 你在这里
├── requirements.txt       ← Python 依赖
├── prep/                  ← Week 8.5: 数据工程准备（Phase 1 前置）
│   ├── clean_pipeline.py          数据清洗管线
│   ├── build_sft_data.py          SFT 数据集构建
│   ├── build_preference_data.py   偏好数据集构建（当前为 stub）
│   └── validate_data.py           数据质量校验
├── week9/                 ← CPT 理论（论文精读）
├── week10/                ← CPT 实战准备（数据 + 配置）
│   └── data_prep_cpt.py
├── week11/                ← CPT 实验（上）
│   └── train_cpt.py
├── week12/                ← CPT 实验（下）+ 灾难性遗忘量化
│   ├── eval_cpt.py
│   └── eval_forgetting.py
├── week13/                ← DPO + GRPO 理论 + 数学推导
│   ├── derivation_dpo_detailed.md
│   └── derivation_grpo.md
├── week14/                ← DPO + GRPO 实战准备
│   └── build_pref_data.py
├── week15/                ← DPO 实验（上）
│   └── train_dpo.py
├── week16/                ← DPO 实验（下）+ 对比 + 失败模式
│   └── compare_dpo.py
├── week17/                ← GRPO 实战（主攻深度）
│   ├── train_grpo.py
│   └── reward_functions.py
├── week18/                ← Distillation 理论
├── week19/                ← Response Distillation 实战
│   └── distill_response.py
├── week20/                ← Feature + On-Policy Distillation（主攻深度）
│   ├── distill_feature.py
│   └── distill_on_policy.py
├── week21/                ← 合成数据生成 + 质量评估
│   ├── self_instruct.py
│   └── evol_instruct.py
├── data/                  ← 共享数据目录
│   ├── raw/
│   └── processed/
├── results/               ← 实验结果
├── notes/                 ← 学习笔记
└── utils/                 ← 共享工具
    ├── eval_benchmark.py
    └── llm_judge.py
```

---

## 深度执行方法论

> 以下三个方法贯穿 Phase 1 全程，每个实验都要遵守。

### 方法一：探索实验跑多个变体，确认实验跑独立 seed

探索阶段至少比较多个预先定义的条件；研究性确认至少使用 3 个独立训练 seed。多个超参臂不能替代重复 seed，点估计 winner 也不能自动称为“最优”。

### 方法二：每次失败都做"尸检"（Post-mortem）

1. 记录现象（不要跳过）
2. 写下 3 个可能原因
3. 设计最便宜的验证实验
4. 修复后记录 trade-off

### 方法三：建立"超参直觉数据库"

```
| 超参 | 值 | 效果 | 意外发现 |
|------|-----|------|---------|
| LoRA rank | 待填 | 记录本项目受控消融结果 | 同时记录 seed 与区间 |
| lr | 待填 | 记录当前模型/数据上的结果 | 不外推为通用最优值 |
```

---

## 验收标准

以下列表同时包含“执行交付”和“研究有效性”。是否通过以 [`audit/exit-gate.md`](audit/exit-gate.md) 为准，不能仅根据周 README 的勾选项判定。

- [ ] 完成至少 1 次严肃的 CPT（有数据混合 ablation）
- [ ] 完成 DPO/GRPO 深度对比（含失败模式），IPO/KTO 仅了解不实验
- [ ] 完成 GRPO 实战（有 domain-specific reward function 设计）
- [ ] 完成至少 2 种蒸馏实验
- [ ] 建立合成数据 pipeline
- [ ] 客观结构化任务有独立 blind test、逐题预测和配对统计；开放式质量声明另有人类盲评，LLM judge 只作补充
- [ ] 至少 3 个独立 insight（你自己发现的 trade-off，不是 paper 里写的）

---

## 必读论文清单

| # | 论文 | 链接 | 阶段 | 优先级 |
|---|------|------|------|--------|
| 1 | Don't Stop Pretraining | https://arxiv.org/abs/2004.10964 | Week 9 | 必读 |
| 2 | HuatuoGPT-II | https://arxiv.org/abs/2311.06750 | Week 9 | 必读 |
| 3 | BloombergGPT | https://arxiv.org/abs/2303.17564 | Week 9 | 必读 |
| 4 | DPO | https://arxiv.org/abs/2305.18290 | Week 13 | 必读 |
| 5 | DeepSeek-R1 (GRPO) | https://arxiv.org/abs/2501.12948 | Week 13 | 必读 |
| 6 | GRPO | https://arxiv.org/abs/2402.03300 | Week 13 | 必读 |
| 7 | IPO | https://arxiv.org/abs/2310.12036 | Week 13 | 背景 |
| 8 | KTO | https://arxiv.org/abs/2402.01306 | Week 13 | 背景 |
| 9 | DistilBERT | https://arxiv.org/abs/1910.01108 | Week 18 | 必读 |
| 10 | Self-Instruct | https://arxiv.org/abs/2212.10560 | Week 18 | 必读 |
| 11 | Zephyr 7B | https://arxiv.org/abs/2310.16944 | Week 18 | 必读 |

---

## 环境准备

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```
