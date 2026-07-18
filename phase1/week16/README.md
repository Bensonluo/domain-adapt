# Week16：DPO 失败模式系统实验 + 跨设定对比

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
| eval 管线 | **参数化 `run_dpo_eval.py`/`eval_winrate.py`**（加 `--sweep/--base/--runs`，默认值保 week15） | week15/16 共用，符合「可切换」铁律；不 fork |
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

**全部在 week12 LoRA 噪声带（±0.04）内 → 无灾难遗忘，安全检查全过。** 连 β=0.01（漂移 434）的 medical Δ 也只有 +0.002——漂移大 ≠ 这次崩医疗，但是个**潜在风险信号**。

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

## 三大失败模式结论（诚实，不美化）

### ① 噪声：训练 acc 单调崩，holdout 不露馅 —— 噪声与过拟合**不可分**

训练 acc 随噪声剂量完美单调下降：1.0 → 0.9 → 0.8 → **0.4**（50% 噪声 ≈ 随机，学不动）。但 holdout 胜率**全线纹丝不动**（meanWR 0.27–0.29，≈ base）。
**判定**：DPO 没有噪声免疫机制——它把能拟合的（含错误标签）都背下来，背不动了（50%）才崩。噪声只体现在训练 acc，**holdout 看不出噪声**，故无法用 holdout 把"数据噪声"和"过拟合"分开。这是 DPO 的一个盲区。

### ② 极端 β：小 β 漂移爆但信号多（危），大 β 强锚不学 —— **修正 week15「β 不可分」**

- **β=0.01**：漂移 **434**（vs control 67，爆炸级），但 holdout meanWR **0.39**（sigmoid 里最高）、matched 0.385。近无 KL → 模型自由漂 → 反而**挖到更多偏好信号**，且这次没崩医疗（Δ+0.002）。代价是漂移极大 = **脆弱/高风险**。
- **β=10**：漂移 13（强锚定），holdout meanWR 0.28 ≈ base —— **几乎没学**。

**判定**：β 在 0.1–0.5（week15 扫的范围）确实不敏感，但**两端不对称且都关键**：太小→漂移爆炸换信号（不稳），太大→僵化不学。week15「β 非关键变量」只在窄区成立，本周**修正**为「β 在极端区决定 稳定性/可学性 trade-off」。

### ③ IPO：长度归一**对症** —— skewed 档首次脱离 0.056 地板（本周唯一正解，带 caveat）

- holdout meanWR **0.45**（base 0.27 / control 0.29，最大跳升）；
- **长度控制档 matched** meanWR **0.462**（control 0.154，**3×**）—— 长度非混淆时，chosen 的 per-token 概率明显更高 = **真偏好信号**（非长度黑客）；
- **长度黑客重灾区 skewed** meanWR **0.167**（control 0.056，**首次脱离地板**）。

**机制**：IPO 优化 mean-logp（per-token），所以 mean-WR 升；但 sum-WR 仍 ≈0（chosen 更长 → Σlogp 必输的结构性劣势**没变**）——**IPO 修的是 mean 不是 sum**，这正是长度归一的精确作用。

**⚠ caveat（诚实）**：IPO 直接优化 mean-logp margin，而 holdout mean-WR 量的就是 mean-logp 胜率 → 部分"涨"来自"IPO 更擅长自己的训练目标"而非"学到更真的偏好"。**最干净的证据是 matched 档**（长度已控制，非目标游戏）：0.154→0.462 说明确实多提取了可迁移的偏好信号。但 matched 只 n=13（0.462≈6/13），**样本小**。结论：IPO **方向对症、量级显著、但需全量数据 + τ 调优确认**（见行动项）。

---

## → week17 行动项（本周结果直接驱动）

1. **IPO 全量 + τ 调优确认**：matched 档 n=13 太小 → 全量 1299 + τ∈{0.3, 0.5, 1.0} 重跑 IPO，确认长度归一收益是否稳定（这是本周最强正信号，值得 stretch）。
2. **IPO 作为 week17 GRPO 的对齐起点**（若全量确认）：长度偏差缓解后的 DPO 模型是更好的 RL 起点。
3. **β=0.01 的高漂移风险**记入 GRPO KL 系数设计（避免重蹈"漂移爆炸"）。
4. **噪声盲区**：week15/16 都用静态离线数据；GRPO 的 on-policy 采样可能对噪声更鲁棒（待验证）。

---

## 验收清单

- [x] 失败模式 run ≥ 2（实际 6：noise×3 + 极端 β×2 + IPO×1，全部从控制基线单变量分支）
- [x] 训练侧信号（noise dose-response 单调 / β=0.01 漂移 434 / β=10 僵化）
- [x] CMMLU 安全检查（全部无灾难遗忘）
- [x] holdout 胜率 + 长度分桶（IPO matched 0.462 / skewed 0.167）
- [x] 跨设定对比报告 [`week16_dpo_comparison.md`](../results/week16_failmode/week16_dpo_comparison.md) + `failure_summary.json`
- [x] 失败模式分析（三大可判定结论，含 IPO caveat）
- [x] IPO/KTO 论文笔记 [`ipo_kto_notes.md`](ipo_kto_notes.md)
- [x] ≥1 个人 insight：**「IPO 修 mean 不修 sum」**——长度归一精确作用于 per-token 胜率，Σlogp 结构性劣势不变；以及 **「β 极端区不对称」**修正了 week15 的窄区结论

---

## 衔接

- **week17**：GRPO（on-policy 独立栈，`beta=0` 不加载 ref，KL 用 k3 estimator）；IPO 全量确认若成立则作对齐起点
- 本周产物：`phase1/results/week16_failmode/{noise_0.1,noise_0.3,noise_0.5,beta_0.01,beta_10,ipo_0.3}{,_fused}/` + domain_gain/forgetting/winrate/failure_summary JSON + week16_dpo_comparison.md
