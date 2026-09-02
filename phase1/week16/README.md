# Week16：DPO 失败模式系统实验 + 跨设定对比

> 方法学纠错见 [`CORRECTIONS.md`](CORRECTIONS.md)。IPO 的现有优势主要出现在与训练目标同构的 mean-logp 指标上，只能作为待确认信号。

> 目标：针对 week15 三大负发现（过拟合 / 长度偏差 / β 不可分），系统跑失败模式 → 三个**可判定**结论。
>
> **思考锚点**："DPO 在什么条件下会失败？失败的表现是什么？"

---

## 关键决策（全部查证，见 plan + [`ipo_kto_notes.md`](ipo_kto_notes.md)）

| 决策 | 选择 | 依据 |
|---|---|---|
| 控制结构 | **全部从 week15 sigmoid β=0.3 baseline 分支，每次只改一个变量** | clean Δ；baseline = `week15_dpo/beta_0.3`（clean, noise=0），不重跑，compare 阶段直接读 |
| IPO 长度归一 | **`--loss-type ipo`，零改码** | 查 TRL v1.8.0 tag 源码 `dpo_trainer.py` L1389-1400：`chosen_scores/chosen_mask.sum()` = mean-logp（源码注释「confirmed with IPO authors」）。直接攻 week15 sum-WR≈0。β 即 τ（0.3 偏低，paper 推荐 0.5–1.0，做受控对比） |
| 数据规模 | **300-subset**（与 week15 baseline 同口径） | 失败模式是**相对信号**，300 足够且 Δ 干净；全量 1299 留 stretch（仅 IPO winner 确认） |
| eval 管线 | **参数化 `run_dpo_eval.py`/`eval_winrate.py`**（加 `--sweep/--base/--runs`，默认值保 week15） | week15/16 共用同一评估入口，通过参数切换配置 |
| 不做 | 人工 50 题 / LLM-judge / 人工长度偏斜数据集 | logprob WR+长度分桶已覆盖对齐信号；LLM-judge 需付费+外发数据（隐私）；长度偏差 week15 已实测 |

---

## 实验设计（6 run，base=`50_50_fused`，limit=300，1 epoch，lr=5e-6，LoRA r16/α32）

| 模式 | run | 改的变量 | 预期 |
|---|---|---|---|
| 噪声剂量 | `noise_{0.1,0.3,0.5}` | 翻转 chosen/rejected | dose-response：acc 能否仍冲 1.0 |
| 极端 β | `beta_{0.01,10}` | β | 0.01→漂移大；10→强锚不动 |
| IPO | `ipo_0.3` | loss=ipo（length-norm） | 攻 sum-WR≈0 长度偏差 |

编排：[`run_failmode_sweep.sh`](run_failmode_sweep.sh)（detached, `nohup`+`caffeinate`）。每 run ~80–180min（300 步 × ~17–20s/step；noise_0.1 因与 eval 并发抢 MPS 拖到 180min——见教训）。

---

## 结果

### ① 训练侧（拟合能力）

| run | β | noise | loss | 末步 margin | acc | 漂移(margin/β) |
|---|---|---|---|---|---|---|
| **control** β=0.3 (wk15) | 0.3 | 0 | sigmoid | 20.19 | **1.00** | 67 |
| noise_0.1 | 0.3 | .1 | sigmoid | 8.99 | 0.90 | 30 |
| noise_0.3 | 0.3 | .3 | sigmoid | 4.87 | 0.80 | 16 |
| noise_0.5 | 0.3 | .5 | sigmoid | **0.10** | **0.40** | 0.3 |
| beta_0.01 | 0.01 | 0 | sigmoid | 4.34 | 1.00 | **434** |
| beta_10 | 10 | 0 | sigmoid | 130.7 | 1.00 | 13 |
| ipo_0.3 | 0.3 | 0 | ipo | 119.9* | 1.00 | — |

*IPO margin 是 mean-logp 尺度，与 sigmoid 的 sum-logp margin **不可直比**。

### ② CMMLU（遗忘检查）

| run | medical_cn Δ | general_cn Δ |
|---|---|---|
| 全部 6 run | −0.008 ~ +0.002 | −0.005 ~ +0.008 |

**全部在本项目经验波动带（±0.04）内，因此当前 development 子集未观察到明显下降。** 这不是“无灾难遗忘”的证明；β=0.01（漂移 434）的 medical Δ 为 +0.002，但漂移仍是潜在风险信号。

### ③ holdout 胜率 + 长度分桶（泛化，100 对）

| run | sumWR | meanWR | matched(n=13) | mid(n=69) | skewed(n=18) |
|---|---|---|---|---|---|
| **base** | 0.01 | 0.27 | 0.154 | 0.348 | **0.056** |
| control β=0.3 (wk15) | 0.02 | 0.29 | 0.154 | 0.377 | 0.056 |
| noise_0.1 / 0.3 / 0.5 | 0.01–0.02 | 0.27–0.29 | 0.154 | 0.35–0.38 | 0.056 |
| beta_0.01 | 0.03 | **0.39** | 0.385 | 0.478 | 0.056 |
| beta_10 | 0.01 | 0.28 | 0.154 | 0.362 | 0.056 |
| **ipo_0.3** | 0.03 | **0.45** | **0.462** | **0.522** | **0.167** |

（全表含 sum_wr/漂移/训练min 见 [`week16_dpo_comparison.md`](../results/week16_failmode/week16_dpo_comparison.md)，由 [`compare_dpo.py`](compare_dpo.py) 生成）

---

## 三类失败模式的结果与分析

### ① 噪声：训练 acc 下降，holdout 胜率变化较小

训练 acc 随噪声剂量增加依次下降：1.0 → 0.9 → 0.8 → **0.4**（50% 噪声时训练准确率接近随机水平）。holdout 胜率的变化较小（meanWR 0.27–0.29，≈ base）。
**分析**：本次配置下，模型能够拟合部分含错误标签的数据，50% 噪声时训练 acc 下降。holdout 胜率对这些噪声设置的变化较小，当前指标尚未区分数据噪声与过拟合各自的影响。

### ② 极端 β：漂移与 holdout 指标的差异 —— 补充 week15 的比较范围

- **β=0.01**：漂移 **434**（vs control 67），holdout meanWR **0.39**（所列 sigmoid 配置中最高）、matched 0.385，medical Δ 为 +0.002。较高的 holdout 指标与较大的漂移同时出现，后续可进一步研究稳定性。
- **β=10**：漂移 13，holdout meanWR 0.28 ≈ base，本次运行的 holdout 改善较小。

**分析**：β 在 0.1–0.5（week15 扫描范围）内的差异较小，本次极端配置呈现不同的漂移与 holdout 表现。此前窄区间结果与本周扩展区间分别记录，用于分析 β 与稳定性、学习效果之间的关系。

### ③ IPO：长度归一配置的结果 —— skewed 档高于 control 的 0.056

- holdout meanWR **0.45**（base 0.27 / control 0.29，为所列配置中的最大增幅）；
- **长度控制档 matched** meanWR **0.462**（control 0.154，**3×**）—— 长度匹配后仍观察到 chosen 的 per-token 胜率较高，为进一步研究偏好信号提供了结果；
- **长度差异较大的 skewed 档** meanWR **0.167**（control 0.056，本次配置下有所提高）。

**机制分析**：IPO 使用 mean-logp（per-token）目标，与 mean-WR 的提高相关；sum-WR 仍 ≈0，序列长度对 Σlogp 的影响仍存在。本次结果中，mean 指标与 sum 指标呈现不同变化。

**解释范围**：IPO 直接优化 mean-logp margin，holdout mean-WR 也采用 mean-logp，二者的关联可能影响指标变化的解释。matched 档在长度匹配后从 0.154→0.462，但仅 n=13（0.462≈6/13），仍需考虑样本波动。全量数据与 τ 对比可进一步研究偏好信号的稳定性（见行动项）。

matched 分桶减少了长度差异，但 mean-WR 仍与优化目标相关；长度匹配与评价指标独立性是两个不同问题。保留这组改善的同时，结合独立任务表现，可以进一步解释偏好训练的实际收益。

---

## → week17 行动项（本周结果直接驱动）

1. **IPO 全量 + τ 调优确认**：matched 档 n=13 太小 → 全量 1299 + τ∈{0.3, 0.5, 1.0} 重跑 IPO，确认长度归一收益是否稳定（这是本周最强正信号，值得 stretch）。
2. **IPO 作为 week17 GRPO 的对齐起点**（若全量确认）：长度偏差缓解后的 DPO 模型是更好的 RL 起点。
3. **β=0.01 的较大漂移**作为 GRPO KL 系数设计的参考（比较不同设置下的漂移与效果）。
4. **噪声盲区**：week15/16 都用静态离线数据；GRPO 的 on-policy 采样可能对噪声更鲁棒（待验证）。

---

## 验收清单

- [x] 失败模式 run ≥ 2（实际 6：noise×3 + 极端 β×2 + IPO×1，全部从控制基线单变量分支）
- [x] 训练侧信号（noise dose-response 单调 / β=0.01 漂移 434 / β=10 的 holdout 改善较小）
- [x] CMMLU development 安全检查（当前子集未观察到明显下降；整体安全性未验证）
- [x] holdout 胜率 + 长度分桶（IPO matched 0.462 / skewed 0.167）
- [x] 跨设定对比报告 [`week16_dpo_comparison.md`](../results/week16_failmode/week16_dpo_comparison.md) + `failure_summary.json`
- [x] 失败模式分析（三大可判定结论，含 IPO caveat）
- [x] IPO/KTO 论文笔记 [`ipo_kto_notes.md`](ipo_kto_notes.md)
- [x] ≥1 项实验观察：**「IPO 修 mean 不修 sum」**——长度归一精确作用于 per-token 胜率，Σlogp 结构性劣势不变；以及 **「β 极端区不对称」**修正了 week15 的窄区结论

---

## 衔接

- **week17**：GRPO（on-policy 独立栈，`beta=0` 不加载 ref，KL 用 k3 estimator）；IPO 全量确认若成立则作对齐起点
- 本周产物：`phase1/results/week16_failmode/{noise_0.1,noise_0.3,noise_0.5,beta_0.01,beta_10,ipo_0.3}{,_fused}/` + domain_gain/forgetting/winrate/failure_summary JSON + week16_dpo_comparison.md
