# Phase 1：深钻阶段 — Deep Dive

> Peng Luo 1-2 年规划第二步 | 13 周 + 数据准备 | 主攻深度：蒸馏 + GRPO
>
> Phase 0 建立了基础（PyTorch / Transformer / LoRA / SFT / 数学推导）。
> Phase 1 在此基础上深钻 4 个方向，补齐短板。

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
│   ├── build_preference_data.py   偏好数据集构建
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
│   └── prepare_preference_data.py
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

### 方法一：每个实验跑 3 个以上变体

不要跑一次就过。"跑通 → 为什么是这个结果 → 换个条件会怎样 → 结论是什么"。

### 方法二：每次失败都做"尸检"（Post-mortem）

1. 记录现象（不要跳过）
2. 写下 3 个可能原因
3. 设计最便宜的验证实验
4. 修复后记录 trade-off

### 方法三：建立"超参直觉数据库"

```
| 超参 | 值 | 效果 | 意外发现 |
|------|-----|------|---------|
| LoRA rank | 8 | 医疗 SFT 够用 | rank 16 没明显提升 |
| lr | 2e-4 | 最优 | 5e-4 loss 震荡 |
```

---

## 验收标准

- [ ] 完成至少 1 次严肃的 CPT（有数据混合 ablation）
- [ ] 完成 DPO/GRPO 深度对比（含失败模式），IPO/KTO 仅了解不实验
- [ ] 完成 GRPO 实战（有 domain-specific reward function 设计）
- [ ] 完成至少 2 种蒸馏实验
- [ ] 建立合成数据 pipeline
- [ ] 所有实验都有严谨评估（benchmark + 人工 + LLM judge）
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
