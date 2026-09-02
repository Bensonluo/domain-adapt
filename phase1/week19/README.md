# Week19：Response Distillation 实战（本地 teacher + 三源受控对照）

> 方法学纠错见 [`CORRECTIONS.md`](CORRECTIONS.md)。三臂结果来自单 seed 和复用 dev；“保知识/稀释”等只作为方向性机制假设。

> 目标：用本地 30B teacher 蒸馏 student，做**受控三对照**（同题只换 completion 来源）干净回答"学 teacher 答案 vs 学人答案，差距在哪"，并与 week17 GRPO 同口径直接比"蒸馏 vs RL"。
>
> **思考锚点**："蒸馏数据训练的小模型，和真实数据训练的，差距在哪？蒸馏的瓶颈是什么？"

---

## 关键决策（探索 + 实测，见 plan）

| 决策 | 选择 | 依据 |
|---|---|---|
| **Teacher** | **Qwen3-30B-A3B-Instruct-2507-MLX-4bit**（MoE 3B-active），**mlx_lm 直跑**（非 LM Studio HTTP） | LM Studio 实测 server-wide 502（见下）；mlx_lm 零 HTTP 依赖、隐私/成本/国内网络全零。与 student 同族（Qwen3）→ tokenizer/风格一致 |
| Student / base | Qwen3-1.7B，base = [`50_50_fused`](../results/week12_lora_cpt/50_50_fused) | 与 DPO/GRPO 同口径，delta 可比 |
| 框架 | TRL 1.8.0 **SFTTrainer** + PEFT-LoRA（r16/α32/dropout0.05/all-linear）+ MPS + bf16 | `completion_only_loss=True` mask prompt token；`peft_config=` 传 Trainer（GRPO 风格，勿 get_peft_model 预 wrap）|
| ★ completion 格式 | 两臂 completion 都**显式格式化** `"{字母}\n{解释}"` | CMExam 人写 `Explanation` 仅 0.1% 首字符是字母（"（B对）（A错）"口诀体）→ `extract_answer` 会抓错；格式化后两臂 eval 口径与 week17 字节一致 |
| Eval（复用） | CMMLU：[`run_dpo_eval.py`](../week15/run_dpo_eval.py)；CMExam holdout：[`eval_cmexam.py`](../week17/eval_cmexam.py) | 同 week15-17 口径。base：CMExam 0.512 / 医 0.5663 / 通 0.6675 |

---

## ⚠️ 三个实测发现（写代码前没想到，跑起来才抓到）

### 1. LM Studio 502 server-wide → 弃 HTTP，改 mlx_lm 直跑

原方案通过 LM Studio serve（`127.0.0.1:1234`）生成数据。本地尝试的模型均返回 502 空响应，耗时约 0.1–0.3s，接口可达但未完成推理。因此改为在 [`generate_teacher_answers.py`](generate_teacher_answers.py) 中直接调用 `mlx_lm.load` 加载一次，再逐题调用 `mlx_lm.generate`，去掉 HTTP 服务依赖，保留断点续跑和增量落盘。

### 2. Gemma MLX thinking 关不掉 → 换 Qwen3-30B-A3B MLX

中途试 Gemma-MLX：`apply_chat_template(enable_thinking=False)` kwarg 被静默吞，模型仍出 `<|channel>thought` 推理块 + tokenizer regex 警告。换 **Qwen3-30B-A3B-Instruct-2507-MLX**：`enable_thinking=False` 干净生效，首字符即字母 + 1-2 句概念解释，与 real 臂密度匹配。

### 3. mlx_lm 0.31.3 API 变更：`temperature` 移出 generate 路径

`mlx_lm.generate(..., temperature=0)` 报 `generate_step() got an unexpected keyword argument 'temperature'`。查已装源码：0.31.3 把采样参数收进 sampler。修复：
```python
from mlx_lm.sample_utils import make_sampler
sampler = make_sampler(temp=0.0)   # greedy = 确定性蒸馏, 取 teacher 众数答案
mlx_lm.generate(model, tokenizer, prompt=..., max_tokens=N, sampler=sampler, verbose=False)
```

---

## 结果（三臂：real / distill / mixed，各 2000 题 SFT，同 base 同 eval）

### ① Teacher 质量（distill 臂天花板）

**teacher 对 gold 准确率 = 0.8645**（2000 题，compliant 100%）。输出样例（字母在前 + 概念解释）：
```
A
冰硼咽喉散具有清热解毒、消肿止痛的功效，常用于口腔溃疡的局部治疗…
```

### ② CMExam holdout（500，全程未训）

| 臂 | correct | accuracy | Δ vs base |
|---|---|---|---|
| base | 256 | 0.512 | — |
| mixed | 262 | 0.524 | +0.012 |
| distill | 265 | 0.530 | +0.018 |
| real | 268 | 0.536 | +0.024 |
| GRPO ref（week17） | — | 0.534 | +0.022 |

**★ 诚实噪声读法**：n=500 配对比较，real−distill = **3 题**，real−mixed = 6 题，配对 SE ≈ 0.032 → **「real > distill > mixed」排序在噪声内，不能当结论**。能站住的只有：**三臂都压过 base**（方向一致，3 个独立信号同向）。即——单论目标任务，**response 蒸馏 ≈ 人写 SFT ≈ GRPO**（都 +0.012~+0.024），分不出高下。`unparseable=0` 三臂都是 → 格式化修正生效，eval 干净。

### ③ CMMLU（development 遗忘检查）—— 方向性信号

| 臂 | medical_cn | Δ | general_cn | Δ |
|---|---|---|---|---|
| base | 0.5663 | — | 0.6675 | — |
| real | 0.5413 | **−0.025** | 0.6625 | −0.005 |
| distill | 0.565 | **−0.001** | 0.675 | +0.008 |
| mixed | 0.5637 | −0.003 | 0.6725 | +0.005 |
| GRPO ref | 0.5687 | +0.0024 | 0.6725 | +0.005 |

**per-task 拆解**：real 臂医学指标下降 −0.025；所列子任务中，**clinical_knowledge 的下降幅度最大**：

| CMMLU 子任务 | base | real | distill |
|---|---|---|---|
| **clinical_knowledge** | 0.55 | **0.45（−0.10）** | 0.53（−0.02） |
| traditional_chinese_medicine | 0.61 | 0.56（−0.05） | 0.61（0） |
| nutrition | 0.61 | 0.57（−0.04） | 0.59（−0.02） |
| virology | 0.64 | 0.61（−0.03） | 0.60（−0.04） |

`clinical_knowledge` 单任务 −0.10 是 real aggregate 下降的主要来源；distill 同任务为 −0.02。这个差异是待验证信号，不能仅凭单 seed 归因给解释来源。

分任务结果用于定位退化集中在哪些能力上。解释来源同时可能改变文本长度、风格和内容，后续可对同题样本的这些属性作比较，分析哪类差异与当前观察有关。

（汇总 [`distill_summary.json`](../results/week19_distill/distill_summary.json)，loss_log / run_config / scores / preds 在 [`phase1/results/week19_distill/`](../results/week19_distill/)）

---

## 结构性解读（核心贡献）

**待验证机制假设：为什么 real 臂下降而 distill 臂未明显下降？** 人写 `Explanation` 与 teacher 输出在风格、长度、信息密度和正确性上同时不同，可能造成不同的知识迁移/干扰；当前设计不能把差异单独归因于“考试口诀体”或“概念解释”。

三条结论：

1. **目标任务点估计接近**：三 SFT 臂与 GRPO 为 +0.012~+0.024；在没有重复 seed 和区间前，只能称当前开发集上接近。
2. **knowledge-preserving 的方向性信号**：real 为医学 −0.025 / 任务 +0.024，distill 为医学 −0.001 / 任务 +0.018。该差异值得受控复验，不能在单 seed 下断言 teacher explanation 必然优于人写解释。
3. **mixed 本次点估计较低**：可能来自风格、采样或优化差异；“两套风格对冲/稀释”只是机制假设。

**teacher/student 差距仍大**：teacher 86% vs student 53%，distill 臂略高于 base。这与“学习了回答模式或部分推理信号”一致，但当前评测不能区分模型究竟学到“怎么答”还是“答什么”。

**与 GRPO 的描述性对照**：GRPO 在该次运行的三个点估计均为正（task +0.022 / 医 +0.002 / 通 +0.005）。这提示其附带退化可能较小，但不能在跨 run 单 seed 比较中确定方法排序。

---

## 验收清单

- [x] 三臂 train+fuse+eval 完成（real/distill/mixed，各 2000 题）→ [`distill_summary.json`](../results/week19_distill/distill_summary.json)
- [x] 受控三对照（同题只换 completion，格式化保证 eval 同口径）
- [x] teacher 准确率报告（0.8645，蒸馏上界 insight）
- [x] 形成待确认机制假设：teacher explanation 可能减少附带退化；mixed 点估计较低的原因待严格控制
- [x] 与 week17 GRPO 同口径直接对比（CMExam holdout）
- [x] 噪声诚实读法（臂间排序在噪声内，三臂压 base 是真信号）

---

## 衔接（week20+）

- **feature / on-policy distillation**（week20 主攻）：本周只 response（学输出），week20 学中间表征 → response-vs-feature 同三对照框架可复用。
- **teacher_answers.jsonl（2000 对）是耐久产物**，week20/21 直接复用，不必重跑 4-7h 生成。
- **N>2000 / 全量**：本周机制验通（三臂一致压 base）；规模留 stretch。
- **更强 teacher / 外部 API**：out（隐私冲突，本地 30B 已够）。

本周产物：`phase1/results/week19_distill/{real,distill,mixed}/`（adapter）+ `{real,distill,mixed}_fused/`（HF）+ `data/`（teacher_answers.jsonl + 四 SFT jsonl）+ `distill_summary.json`；代码 [`generate_teacher_answers.py`](generate_teacher_answers.py)/[`prepare_distill_data.py`](prepare_distill_data.py)/[`train_distill_sft.py`](train_distill_sft.py)/[`summarize_distill.py`](summarize_distill.py)/[`run_distill.sh`](run_distill.sh)。
