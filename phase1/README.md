# Phase 1：深钻阶段 — Deep Dive

> 第二步 | 13 周 + 数据准备 | 主攻深度：蒸馏 + GRPO
>
> 在 Phase 0 的实现与实践基础上，深入 CPT、偏好优化、蒸馏与合成数据。

## 阶段成果

Week 9–21 已形成理论推导、训练流程、多组对比实验和失败复盘。后续修订完成了 grouped preference split、评估数据登记、历史结果重分析和五类实验设计。

本阶段重点展示方法选择、实际结果以及发现问题后的调整。既有 CMExam 500 题与反复使用的 CMMLU 子集用于 development comparison；新准备的确认数据和多 seed 设计尚未运行。

- 声明—证据矩阵：[`audit/claim-evidence-matrix.md`](audit/claim-evidence-matrix.md)
- 结果解读与改进方向：[`audit/exit-gate.md`](audit/exit-gate.md)
- 后续实验建议：[`audit/rerun-plan.md`](audit/rerun-plan.md)
- 交付物清单：[`artifact_manifest.md`](artifact_manifest.md)
- 机器可读证据注册表：[`artifact_registry.json`](artifact_registry.json)
- 评估隔离与盲测协议：[`audit/blind-test-protocol.md`](audit/blind-test-protocol.md)
- Benchmark 污染登记：[`audit/benchmark_registry.json`](audit/benchmark_registry.json)
- 结果索引说明：[`audit/ADR-001-evidence-registry-and-blind-test.md`](audit/ADR-001-evidence-registry-and-blind-test.md)
- 审计运行入口：[`audit/README.md`](audit/README.md)
- 五类后续实验设计：[`confirmation/README.md`](confirmation/README.md)

| 周 | 已交付内容 | 实践中的发现与调整 |
|---|---|---|
| W8.5 | 数据清洗与校验准备 | 实际数据路线由 W10/W14 承接；偏好构建 stub 未继续扩展 |
| W9 | CPT 理论笔记 | 区分论文中的条件与本项目适用范围 |
| W10 | 真实语料处理管线与配置 | 实际采用约 1414 万 token，未扩展到原定 1B–3B |
| W11 | demo CPT 训练 | 验证训练流程，后续以真实语料继续实验 |
| W12 | 三比例 LoRA-CPT 对比 | 50/50 作为后续基线；收益来源仍有多变量混杂 |
| W13 | DPO/GRPO 数学推导 | 连接目标函数与训练实现 |
| W14 | 偏好数据与 grouped split | 发现历史 50 组 prompt 交叉；新切分 overlap=0 |
| W15 | DPO beta sweep | 小 matched bucket 中差异不明显，未据此确定最优 beta |
| W16 | DPO 失败模式与 IPO 对比 | 分析长度偏差及目标相关指标 |
| W17 | GRPO 训练、评估与重分析 | 发现 8/500 题训练重叠；clean delta 范围为 +0.61pp～+3.86pp，缺 450 条逐题预测 |
| W18 | 蒸馏理论笔记 | 比较不同蒸馏形式的前提和取舍 |
| W19 | response distillation 三臂实验 | teacher explanation 出现积极信号，机制解释仍是研究假设 |
| W20 | logit KD 与 rejection sampling 多臂实验 | 形成配置比较及完整度较高的 lineage |
| W21 | 合成生成、质检、替代实验与 clean matched 数据 | 点估计接近 control，区间尚不支持非劣效；清理后数据尚未重训 |

---

## 总览

围绕四个方向展开：

1. **Continual Pre-training (CPT)** — 通过领域语料继续预训练
2. **DPO / GRPO / Preference Tuning** — 对齐模型偏好
3. **Knowledge Distillation** — 把大模型能力迁移到小模型
4. **Synthetic Data Generation** — 用数据工程放大训练效果

**主攻深度方向**：蒸馏 + GRPO

---

## 目录结构

```
phase1/
├── README.md              ← 阶段概览
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

实验记录主要关注比较条件、失败原因和参数取舍。

### 方法一：探索实验跑多个变体，确认实验跑独立 seed

多个变体有助于寻找候选配置；独立 seed 用于观察训练波动，两者回答不同问题。当前探索结果已形成后续设计，其中采用 3 个 seed 进一步比较稳定性。

### 方法二：从失败现象找到下一步

记录现象和可能原因，优先尝试低成本的区分实验，再总结修改后的效果与取舍。每周复盘保留这一思考过程。

### 方法三：记录超参数的实际取舍

```
| 超参 | 值 | 效果 | 意外发现 |
|------|-----|------|---------|
| LoRA rank | 待填 | 记录本项目受控消融结果 | 同时记录 seed 与区间 |
| lr | 待填 | 记录当前模型/数据上的结果 | 不外推为通用最优值 |
```

---

## 交付概览

- [x] 小规模真实语料 CPT 与数据混合比例对比。
- [x] DPO/GRPO 理论、实践及失败模式分析；IPO 实验与 KTO 阅读。
- [x] GRPO 训练与 domain-specific reward 设计。
- [x] response distillation、logit KD 与 rejection sampling 实验。
- [x] 合成数据生成、质量检查与替代实验。
- [x] 结果重分析、数据切分修订和每周纠错记录。

后续实验尚未执行的部分单独列在 [实验建议](audit/rerun-plan.md)，不与已有交付混列。

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
