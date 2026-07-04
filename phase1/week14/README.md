# Week 14：DPO + GRPO 实战准备

> 目标: 构建偏好数据集，阅读 TRL 源码，设计实验矩阵。
> 预计时间: 12-16 小时

> **思考锚点**: "偏好数据的质量怎么保证？chosen 真的比 rejected 好吗？"

> **定位**：衔接 week13（DPO/GRPO **理论**）→ week15（实际跑 DPO 训练）。本周**不跑任何训练**，只做数据 + 源码阅读 + 实验设计三件准备。详见 [`experiment_matrix.md`](experiment_matrix.md)。

---

## 本周做了什么（实际执行）

三件准备全部交付，**按真实约束（0.8B / Apple Silicon / 已有资产）校准**，不照搬课程 plan 文本里 3B 的假设：

1. **偏好数据集 + QC**（[`build_pref_data.py`](build_pref_data.py)）—— 用公开 in-domain 数据集（魔搭 `medical_evidence_DPO`）替代 prep 阶段 stub 的 teacher/student 生成。**免 teacher API、免下载大模型**。1399 对干净 `{prompt, chosen, rejected}`，QC 发现**严重长度偏差**（chosen 更长 93.5%）。
2. **TRL 源码阅读**（[`trl_source_notes.md`](trl_source_notes.md)）—— 读 `dpo_trainer.py` + `grpo_trainer.py` main 分支，补全 week13 的「为什么」→「代码怎么实现」。确认 GRPO **无 critic**、找到 reward 归一化的真实代码、IPO 的 length-normalized loss（直接对应本周长度偏差发现）。
3. **实验矩阵 + SFT 决策**（[`experiment_matrix.md`](experiment_matrix.md)）—— 6 行实验按 0.8B 校准；把「Phase 1 从未跑过 SFT 但矩阵假设 CPT+SFT 基线」这一最大课程缺口变成显式决策（方案 A CPT-only 默认 / 方案 B 先补 SFT 备选）。

---

## Day 1-2: 偏好数据集构建

### 数据源（替换 stub）

prep 阶段 [`build_preference_data.py`](../prep/build_preference_data.py) 是**纯 stub**（`generate_preference_pairs` 函数体 `pass`，main 只 print TODO），且依赖不存在的 `phase1/data/raw/questions.jsonl` + teacher/student 大模型生成（付费 API / 下载大模型）。本周用**公开数据集替代**：

- **魔搭 `modelzhang/medical_evidence_DPO`**（中文医疗循证 DPO）—— in-domain 匹配 CPT 医疗语料，魔搭国内可达（memory [[modelscope-qwen-download]]），专家生成 `{prompt, chosen, rejected}` 三元组，免 teacher API。
- teacher/student 生成（`deepseek_vs_qwen6B_DPO.jsonl`，2099 条 chat-message 格式）作为**已考虑的备选**记录在 QC 报告里，不执行（数据集已自带预生成版本）。

### 跑

```bash
# 1. 拉数据集（魔搭 git clone with LFS，国内可达）
git clone https://www.modelscope.cn/datasets/modelzhang/medical_evidence_DPO.git \
    phase1/data/raw/preference_src/medical_evidence_DPO

# 2. 构建 + QC（过滤/去重/token 统计/抽检一条龙）
python phase1/week14/build_pref_data.py
```

产物：
- [`phase1/data/processed/preference/train.jsonl`](../data/processed/preference/train.jsonl)（1399 对，gitignored）
- [`pref_qc_report.json`](pref_qc_report.json)（QC 统计）
- [`sampled_100.jsonl`](sampled_100.jsonl)（100 对人工抽检）

### QC 关键发现（驱动实验设计）

- **长度偏差严重**：chosen 更长占 **93.5%**，`|chosen−rejected|/max > 0.5` 占 **32%**。→ **长度黑客风险高**：DPO 用 `Σ logp`，chosen 更长 → logp 总和更大 → 模型可能学「更长=更好」而非「更好=更好」。必须配长度控制评估 + 考虑 IPO / length-normalized DPO（见矩阵行 2 可选第 7 实验 + TRL 笔记 §1.5）。
- **max_seq 规划**：prompt+chosen token p95=2980 / p99=3356 / max=3815 → DPO `max_length≈4096`（仅 54% 对 fit 2048）。0.8B 全量 + 4096 ctx 在统一内存上偏紧，需 grad-checkpoint。
- 去重策略修正：按 **(prompt, chosen, rejected) 完整三元组**去重（不按 prompt）——同一 prompt 带不同 chosen/rejected 是合法偏好对（更多 preference signal），不应丢。最终 0 条完全重复。

---

## Day 3-4: TRL 源码阅读

读 `huggingface/trl` main 分支（不本地装 torch/TRL，week15 才需要）。详见 [`trl_source_notes.md`](trl_source_notes.md)。核心收获：

**DPO**：
- 老版教程里的 `dpo_loss` / `get_batch_logps` 方法**已不存在**，所有逻辑 inline 进 `_compute_loss` (L1321)。
- per-token logprob 用 `selective_log_softmax` + `completion_mask` 屏蔽 + `sum(dim=1)`，batch 顺序写死 `[chosen, rejected]`。
- loss = `-F.logsigmoid(beta * (chosen_logratios - rejected_logratios))`，与 week13 公式逐项对应。
- reference 三选一：预计算 / PEFT-disable-adapter / 独立 ref 模型。
- **IPO 用 length-normalized score**（除以 completion token 数）—— 直接缓解本周长度偏差发现。

**GRPO**：
- `RepeatSampler` 把每个 prompt 连采 G 次；reward 归一化 `advantages = (rewards − group_mean) / (group_std + 1e-4)`，`rewards.view(-1, num_generations)` 是核心。
- **确认无 critic**（grep `critic|value_head|value_network` 全空）—— baseline 就是 group mean，这是 GRPO 对 PPO 的核心简化。
- KL 用 Schulman k3 estimator `exp(Δ) − Δ − 1`；**`beta=0` 时根本不加载 ref 模型**（L800-801），0.8B 显存关键。
- PPO clipped surrogate loss + per-token KL 加到 loss 上。

---

## Day 5: 实验矩阵设计

6 行实验，按 0.8B 真实约束校准（不照搬 plan 里 3B 的 batch/reward 假设）。详见 [`experiment_matrix.md`](experiment_matrix.md)。

| # | 基线 | 方法 | 关键超参 | 预期信号 |
|---|------|------|----------|----------|
| 0 | A:CPT-only / B:CPT+SFT | —（对照组） | — | 基线分；DPO/GRPO 的 delta 都相对它 |
| 1 | 同上 | DPO | β=0.1（弱） | 轻微偏好提升，遗忘最小 |
| 2 | 同上 | DPO | β=0.3（中） | 对齐/遗忘甜点区候选 |
| 3 | 同上 | DPO | β=0.5（强） | 偏好提升最大但医疗可能崩 |
| 4 | 同上 | GRPO | rule reward（医学关键词+长度惩罚+结构分） | 验证 GRPO 在 0.8B 可跑 |
| 5 | 同上 | GRPO | reward ablation（去长度惩罚/换纯格式） | 量化 reward 各项贡献，**直接验证长度黑客** |

**SFT 缺口决策**（本周最大产出之一）：Phase 1 从未跑过 SFT，但矩阵 + week15 都假设 CPT+SFT 基线。方案 A（推荐，默认）首轮用 **CPT-only 基线**（`real_cpt_fused` 已 fuse-ready，立即可跑）；方案 B（备选）week15 先补 SFT 再 DPO。决策权留给进入 week15 时定。

**评估**复用 [`week12/_eval_core.py`](../week12/_eval_core.py)：CMMLU `medical_cn`/`general_cn`（领域保留 + 通用遗忘）+ 偏好胜率（logprob，无 LLM judge）+ **长度控制胜率**（长度黑客检测，本周 QC 驱动）。

---

## 交付物

- [x] 偏好数据集（1399 对）+ QC 报告 — [`build_pref_data.py`](build_pref_data.py) / [`pref_qc_report.json`](pref_qc_report.json) / [`sampled_100.jsonl`](sampled_100.jsonl)
- [x] TRL 源码阅读笔记 — [`trl_source_notes.md`](trl_source_notes.md)
- [x] 实验设计文档 — [`experiment_matrix.md`](experiment_matrix.md)

---

## 验收清单

- [x] 偏好数据构建完成（1399 对，in-domain 公开数据集替代 stub）
- [x] 长度偏差**已量化并可控**（chosen 更长 93.5% → 矩阵配长度控制胜率 + IPO/length-normalized 可选实验）
- [x] DPOTrainer + GRPOTrainer 源码阅读完成（DPO loss + GRPO reward normalization 各有「代码做了什么 + 对应 week13 哪个公式」段）
- [x] GRPO reward function 设计完成（矩阵行 4/5：医学关键词 + 长度惩罚 + 结构分，含 ablation）
- [x] 实验矩阵文档完成（6 行，每行可执行 + 基线 + 评估路径）
- [x] SFT 缺口显式决策（方案 A 默认 / B 备选，理由清楚）

---

## 实际执行偏离记录（对照课程 plan）

| plan 假设 | 实际 | 处理 |
|-----------|------|------|
| prep `build_preference_data.py` 跑 teacher/student 生成 | 纯 stub + 无 questions.jsonl | 用公开数据集 `medical_evidence_DPO` 替代，免 API |
| 基座 3B | 真实 `Qwen3.5-0.8B-Base-ms` | 矩阵 batch/reward/显存按 0.8B 校准 |
| 矩阵基线 = CPT+SFT | Phase 1 从未跑 SFT | 显式决策方案 A(CPT-only)/B(先 SFT)，不擅自扩范围 |
| 本地装 TRL/torch | 国内 + Apple Silicon + 偏重 | 本周纯 GitHub raw 读源码，week15 才装 |
| `mlx_lm` 跑 DPO | `mlx_lm` 无 DPO 子命令（已确认） | 主路线 TRL，MLX-DPO footnote 备选 |

---

## 不在本周范围（留 week15+）

- 实际跑 DPO/GRPO 训练（week15，`phase1/week15/train_dpo.py` 已存在）
- 本地装 torch/TRL（week15 才需要）
- 跑 SFT（本周**决策**，week15 视方案 A/B 执行）
- teacher/student 偏好生成（public 数据集替代，备选记录在 QC 报告）
- GRPO reward function 的真实实现（本周只在矩阵里设计，week15/16 跑）
