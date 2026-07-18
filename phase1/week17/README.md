# Week17：GRPO 实战（MCQ 答对率 reward，on-policy RL 落地）

> 目标：把 GRPO stub 填成能跑的 on-policy RL，产出 **GRPO vs DPO 对比** + reward hacking 诊断。
>
> **思考锚点**："reward 上升但质量真的变好了吗？怎么区分真实提升和 reward hacking？" —— 本周用**客观 MCQ 答对率**reward，让这个问题几乎自动回答（答对就是答对，hacking 几乎不可能）。

---

## 关键决策（全部查证 + 实测，见 plan + [`phase1/week14/trl_source_notes.md`](../week14/trl_source_notes.md)）

| 决策 | 选择 | 依据 |
|---|---|---|
| reward | **MCQ 答对率**（用户决策） | 开放式医疗 reward 区分度弱（G 个 completion 格式近似→advantage≈0→学不动）且易 hacking；MCQ 客观、区分度天然、hacking 几乎不可能 |
| base | [`50_50_fused`](../results/week12_lora_cpt/50_50_fused)（**非 stub 过时的 week11_cpt_pure**） | 与 DPO（week15/16）同口径，delta 可比 |
| 数据 | **CMExam**（[`fzkuji/CMExam`](https://huggingface.co/datasets/fzkuji/CMExam)，68K 简体医学选择题）→ 8K train + 500 holdout（test split，全程未训） | 小数据 hf-mirror 可下；纯简体；客观答案可校验 |
| loss | **`dapo`**（TRL v1.8.0 默认） | 自带长度偏差消除（呼应 week16 IPO）；源码 grpo_trainer.py L2949 |
| KL | **`beta=0`**（不加载 ref，L888-894，省 3.4GB） | GRPO 锚是 group baseline（advantage=(R−mean)/std，L2394-2418），不是 KL；beta=0 是 TRL 默认 |
| QLoRA | **`peft_config=` 传 Trainer，不自己 wrap**（L411-416 预 wrap 会报错） | 源码确认，与 DPO 不同 |
| 栈 | **TRL+MPS**（非 MLX-GRPO） | smoke 实测 MPS 不崩（见下"风险解除"）|

---

## ⚠️ 三个实测发现（写代码前没想到，跑 smoke/learncheck 才抓到）

### 1. MPS 风险解除（plan 里两 agent 证据矛盾，实测裁断）

plan 阶段两 agent 冲突：悲观方举 TRL [#4692](https://github.com/huggingface/trl/issues/4692)（M4 Max GRPO `mps_matmul` LLVM crash）+ PyTorch [#180776](https://github.com/pytorch/pytorch/issues/180776)（MPS bf16 `F.linear` 静默坏值，报告者点名 GRPOTrainer），公开成功案例全是 MLX-GRPO。**DPO 先例不成立**（DPO 无 generation rollout，GRPO 每步 on-policy generate）。

**smoke 实测（max_steps=2）裁断**：不崩、reward 非 NaN（0.5）、`frac_reward_zero_std=0`（每组有对有错=真 advantage 信号）→ **TRL+MPS+GRPO 在本机可跑，不需切 MLX**。环境兜底 `MTL_TIMEOUT=0`+`PYTORCH_ENABLE_MPS_FALLBACK=1` 仍保留。

### 2. Qwen3 是 thinking 模型 → completion 不终止（改 prompt 预算）

smoke 发现 `clipped_ratio=1`、`mean_terminated_length=0`：所有 completion 撞 256 上限，无 EOS。直接 generate 探针定位三种格式：
- **RAW `…答案：`** → 首字符即答案字母（"C 2.下列…"），其后 ramble 编新题，不终止 ✅ reward 可用
- chat 模板 → 触发 `<think>` 长推理，256 token 跑不完 ❌
- chat + `enable_thinking=False` → 输出冗长解释，不出字母 ❌

**结论**：RAW 格式首字符即答案（reward 抓首字母），ramble 不影响 reward → `max_completion_length 256→48`（答案=token#1，48 够捕获，比 256 快 ~5×，dapo 已长度归一）。

### 3. `lr=1e-6` 不学 → `1e-5`（grad_norm 健康但零漂移）

learncheck（50 步）`lr=1e-6`：`grad_norm≈1.2` 健康、advantage 正确（±0.5/1.5），但 reward **零趋势**。算账：50 步 LoRA 漂移 ~5e-5 ≈ adapter 幅度 0.1%，太弱。LoRA-GRPO 文档范围 1e-5~2e-5（LoRA 需比 full-FT 高 lr，同 DPO 用 5e-6 之理）。换 `1e-5` → reward 0.383→0.562 干净上升。

> **副产物**：`loss≈1e-9` 是 **dapo 归一化假象**，不是 bug。插桩 `_compute_loss` 确认：dapo 把 per-token loss 求和后除以全局 token 数，而 advantage 是组内去均值的（同组 Σadvantage=0），on-policy `coef_1≈1` 时求和≈0；但**梯度** `A_t×∇logπ` 非零（policy gradient 定理），模型照常更新。reward 上升证实。

---

## 结果（mcq_base：lr=1e-5, G=4, beta=0, loss=dapo, max_completion_length=48, 1500 步）

### ① 训练侧（CMExam train 8K，reward 曲线）

| 训练段 | 前25% | 25–50% | 50–75% | 后25% |
|---|---|---|---|---|
| mean reward | 0.454 | 0.479 | **0.557** | 0.551 |

**train reward +0.097（0.454→0.551），50% 后 plateau** —— 学到了，但撞能力天花板。

### ② CMExam holdout 答对率（500，全程未训，GRPO 新增指标）

| 模型 | accuracy | unparseable |
|---|---|---|
| base (50_50_fused) | 0.512 | 0% |
| **GRPO mcq_base** | **0.534** | 0% |
| **Δ** | **+0.022（+2.2pp）** | — |

**真迁移**：train +0.097 → holdout +0.022（train > holdout = 对训练子集轻度过拟合，但 holdout 增益确为正，非纯记忆）。`unparseable=0` 两边都是 → reward 设计健全，模型乖乖出字母。

### ③ CMMLU（遗忘检查）

| | medical_cn | general_cn |
|---|---|---|
| base | 0.5663 | 0.6675 |
| GRPO | 0.5687 | 0.6725 |
| **Δ** | **+0.0024** | **+0.005** |

**全在 week12 LoRA 噪声带（±0.04）内 → 无灾难遗忘**（且略正）。GRPO 对齐 MCQ 格式没给医疗知识加成（意料中），但也没破坏。

（汇总 [`grpo_summary.json`](../results/week17_grpo/grpo_summary.json)，原始 loss_log / run_config / scores 在 [`phase1/results/week17_grpo/`](../results/week17_grpo/)）

---

## GRPO vs DPO 对比（结构性，非单指标）

DPO（week15/16）和 GRPO 优化的是**不同信号**，不能压成单一指标比大小，按轴对比：

| 轴 | DPO（week15/16） | GRPO（week17） |
|---|---|---|
| **信号类型** | logprob 胜率（chosen vs rejected，**代理**信号） | MCQ 答对率（**客观**，答对即答对） |
| **信号强度** | 弱：sumWR≈0（长度偏差，chosen 更长必赢 Σlogp） | 强且干净：train reward +0.097，holdout +0.022 |
| **数据需求** | 偏好对（chosen/rejected，1399 对） | prompt + reward 函数（8000 MCQ） |
| **机制** | 离线对比（无 generation） | on-policy generation + group baseline |
| **遗忘** | medical Δ −0.006（week15 β=0.3） | medical Δ **+0.002**（略正） |
| **holdout 泛化** | meanWR 0.29（β=0.3）/ 0.45（ipo，带目标泄漏 caveat） | 答对率 +2.2pp（无 caveat，客观） |
| **主要风险** | 长度偏差（week16 三大发现之一） | reward hacking（MCQ 客观 → 几乎不可能，本周 unparseable=0 证实） |
| **beta/KL 作用** | β 是核心（week16 极端 β 不对称） | beta=0 默认不加载 ref（group baseline 是锚） |

**判定（诚实）**：GRPO 在**信号干净度**上明显胜出（客观 reward → 无长度偏差、无目标泄漏、holdout 增益可信）；在**绝对增益幅度**上 modest（+2.2pp）——MCQ 答对率 reward 的天花板是模型**已有知识**，RL 能 sharpen 选择但不能凭空加知识。DPO 的 holdout 信号被长度偏差/目标泄漏污染，不可直接比大小，但其**偏好对齐**目标与 GRPO 的**答案正确性**目标本就不同。

---

## 验收清单

- [x] GRPO 训练跑通（TRL+MPS+PEFT，smoke 风闸 + 1500 步正式跑）
- [x] Domain-specific reward function（MCQ 答对率，正确签名，10 单测过）
- [x] GRPO vs DPO 完整对比（结构性 8 轴表）
- [x] reward hacking 诊断：**未出现**（客观 reward + unparseable=0 + holdout↑ + 无遗忘 = 真提升）
- [x] ≥1 个人 insight：**「loss≈0 ≠ 不学习」**（dapo 归一化假象 vs policy gradient 非零梯度的区分）+ **「MCQ reward 天花板=模型知识」**（RL sharpen 选择不增知识）+ **「DPO 先例对 GRPO 不成立」**（generation rollout 路径才触发 MPS corruption）
- [x] CMExam 数据闭环（68K 下载 → 8K/500 切分 → train → holdout 客观 eval）

---

## 衔接（week18+ stretch）

本周机制验通（GRPO on MPS 可跑、能学、能迁移、无遗忘）。**不在范围**（留 stretch）：
- **thinking-mode GRPO**：本周关掉 thinking（RAW 短 completion）。开 thinking + 大预算（max_completion_length 1024+）让模型推理后答题 —— 更贴 Qwen3 本性，可能突破 +2.2pp 天花板（RL 强化推理链而非仅首字母）。
- **ablation sweep**：num_generations 4→8/16（advantage 估计更稳）、temperature（降 `frac_reward_zero_std`，本周 0.4-0.6 浪费）、beta>0（加 KL 锚，验 week16 β 教训在 GRPO 是否复现）。
- **60K 全量**：本周 8K 子集验机制；全量 + 更多步可能推高 plateau（本周 50% 后就 plateau，故优先级低于 thinking-mode）。
- **model-based reward**：out（付费 + 外发数据违反隐私）。

本周产物：`phase1/results/week17_grpo/{mcq_base,mcq_base_fused,base_hf,base_cmexam_holdout.json,grpo_summary.json,domain_gain.json,forgetting.json}` + 代码 [`train_grpo.py`](train_grpo.py)/[`reward_functions.py`](reward_functions.py)/[`prepare_cmexam.py`](prepare_cmexam.py)/[`eval_cmexam.py`](eval_cmexam.py)/[`run_grpo.sh`](run_grpo.sh)。
