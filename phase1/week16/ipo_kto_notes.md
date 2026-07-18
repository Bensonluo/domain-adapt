# Week16 Day5：IPO / KTO 论文要点

> 服务于 `week16_dpo_comparison.md` 的「why IPO」段。IPO 是本周失败模式实验的核心 (攻 week15 长度偏差),
> KTO 作对比理解 (本周不跑: 需非配对数据, 我们的数据是配对 chosen/rejected)。

---

## IPO — Identity Preference Optimization (Azar et al. 2023)

论文: [arXiv:2310.12036](https://huggingface.co/papers/2310.12036) "A General Theoretical Paradigm to Understand Learning from Human Preferences"

### 核心思想
把「偏好学习」抽象成: 学一个 reward 差距到**固定目标 margin** 的回归, 而非 DPO 那种 logistic「拉开 chosen/rejected」。
直接对 DPO 的过拟合敏感 (DPO loss 对 reward 差呈指数饱和, 易把 margin 推到无穷 → 过拟合); IPO 用平方损失, margin 到 1/(2τ) 就「满足」, 不再无限推。

### 损失 (Eq.17, τ 为正则强度)
```
L_IPO = ( (mean_logp_ratio_chosen − mean_logp_ratio_rejected) − 1/(2τ) )²
```
关键: IPO 的 score 是**长度归一**的均值 log-ratio, 不是 DPO 的求和 Σ log-ratio。

### TRL v1.8.0 实测 (本周 IPO run 走的就是这条) — 已查 v1.8.0 tag 源码
源码 `dpo_trainer.py` L1389-1400:
```python
elif loss_type == "ipo":
    chosen_avg_score   = chosen_scores   / chosen_mask.sum(dim=1).clamp(min=1.0)   # mean-logp, 长度归一
    rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
    ipo_delta = chosen_avg_score - rejected_avg_score
    per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2                     # β 即 τ
```
源码注释原文: *"IPO uses sequence-level log-prob differences; in code these are token-summed over the completion, which makes the squared loss scale with completion length. We therefore normalize by the number of completion tokens ... not explicitly discussed in the IPO paper; we confirmed this choice with the IPO authors."*

→ **TRL 的 IPO 确实长度归一** (mean-logp)。一个网络摘要称「IPO 不长度归一、要 SimPO」对 TRL 是**错的** (可能混淆旧版/其他库, 见 [trl#2964](https://github.com/huggingface/trl/issues/2964) 历史)。以 v1.8.0 源码为准。

### 超参
- **无 `ipo_tau` 字段**: TRL 里 β 就是 τ ([dpo_config.py](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/dpo_config.py) L103-106)。
- τ 范围: IPO paper 常用 0.5–1.0; 旧 TRL v0.8.1 文档建议 0–0.5。本周取 **β=0.3** (与 sigmoid β=0.3 baseline 单变量对照), 偏低端 — 若 IPO 不显效, τ 是首要待查混淆。

### 为什么本周选 IPO (不是 SimPO/KTO)
week15 实测: sigmoid DPO 的 **Σ-logp 目标**让「chosen 更长 → Σ logp 必更负 → 必输」(sum-WR≈0, skewed 档 mean-WR 0.056)。这是 DPO 文献经典长度黑客。IPO 的 **mean-logp score** 消掉 Σ-长度混淆 → 直接检验「长度偏差是不是 loss 形式问题」。SimPO 也长度归一但无 reference model (我们的栈要 ref), KTO 要非配对数据 (我们是配对)。IPO 是对症且数据兼容的唯一选项。

---

## KTO — Kahneman-Tversky Optimization (Ethayarajh et al. 2024)

论文: [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) "KTO: Model Alignment as Prospect Theoretic Optimization" (ICML 2024, 1047+ 引用)
项目页: [winniexu.ca/research/kto](https://winniexu.ca/research/kto)

### 核心思想
HALO (Human-Aware Loss Objective) — 用 prospect theory (Kahneman-Tversky) 的效用函数建模人对「收益 vs 损失」的不对称感知 (loss aversion: 损失权重 > 等量收益), 直接最大化生成效用, 而非最大化 chosen-vs-rejected 似然差。

### 与 DPO 的关键区别
| | DPO / IPO | KTO |
|---|---|---|
| 数据 | **配对** (chosen, rejected) | **非配对** (单输出 + desirable/undesirable 标签) |
| 理论 | reward margin (DPO) / 平方回归 (IPO) | prospect theory 效用 (KT 函数) |
| 损失厌恶 | 不建模 | 显式建模 (λ 加权负例) |
| 数据获取 | 难 (要人工排序出胜负对) | 易 (点赞/点踩即可) |

### 实践意义
- 只需 binary signal (单条好/坏), 不需 curated 配对 → 真实场景 (用户反馈) 更易收集。
- 论文称 match/exceed DPO。
- **本周不跑**: 我们的数据 (medical_evidence_DPO, 1399 配对) 是配对格式, KTO 要非配对; 且 `train_dpo.py` 的 `--loss-type` choices 是 {sigmoid, ipo, hinge, robust} (无 kto, DPOTrainer 走配对)。KTO 在 TRL 是独立 `KTOTrainer` + 非配对数据格式。留作后续。

---

## 本周实验定位
- **IPO (ipo_0.3)**: 唯一实测的 length-norm loss, 头号看点 = sum-WR / skewed 档 mean-WR 是否脱离 week15 sigmoid 的 ≈0 / 0.056。
- **KTO**: 理解性阅读, 不实验 (数据格式不匹配)。

**Sources:**
- IPO: [arXiv:2310.12036](https://huggingface.co/papers/2310.12036) | [TRL v1.8.0 dpo_trainer.py](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/dpo_trainer.py) | [TRL v1.8.0 dpo_config.py](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/dpo_config.py) | [trl#2964](https://github.com/huggingface/trl/issues/2964)
- KTO: [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) | [ICML/ACM](https://dl.acm.org/doi/10.5555/3692070.3692574) | [项目页](https://winniexu.ca/research/kto) | [HF papers](https://huggingface.co/papers/2402.01306)
