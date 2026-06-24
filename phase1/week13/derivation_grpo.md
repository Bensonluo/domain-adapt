# GRPO 机制整理 + 与 DPO 对比

> **一句话**：GRPO（Group Relative Policy Optimization）是 PPO 的精简版——**砍掉 critic 网络，用一组采样的 reward 均值/方差当 baseline 估 advantage**。
> 它保留了 RL 的在线采样，但工程上轻量得多，是 DeepSeek-R1 reasoning 涌现的核心引擎。
>
> 配套论文：Shao et al. 2024, *DeepSeekMath* §4 (arXiv:2402.03300)；DeepSeek-R1 (arXiv:2501.12948)

---

## ⚠️ 先纠一个常见误解（README 原答案错了）

> "GRPO 不需要 reference model" —— **错**。
>
> 原文 §4.1.1 白纸黑字："GRPO regularizes by directly adding the **KL divergence between the trained policy and the reference policy** to the loss"。
>
> **正确**：GRPO **保留 reference model**（做 KL penalty）。它砍掉的是 **value/critic function $V_\psi$**（PPO 必须的），用 group baseline 代替。
> DPO 也保留 reference model。两者的 reference model 角色不同（见 §5），但**都在**。

---

## 1. 起点：PPO 在 RLHF 里长什么样

PPO 的目标（带 KL 约束的 reward 最大化）：

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}\bigg[\frac{1}{|o|}\sum_t \min\!\Big(\rho_t\hat{A}_t,\;\text{clip}(\rho_t,1\!-\!\varepsilon,1\!+\!\varepsilon)\hat{A}_t\Big) - \beta\,\mathbb{D}_{KL}[\pi_\theta\|\pi_{\text{ref}}]\bigg]
$$

- $\rho_t = \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}$：重要性采样比（off-policy 校正）
- $\hat{A}_t$：**advantage**，PPO 里用 GAE 算，**需要一个 value function $V_\psi$**
- KL 项把策略拉回 reference

**PPO 的两个重负担**（GRPO 要砍的）：
1. **Value/critic 网络 $V_\psi$**：和 policy 同大小的额外模型，显存翻倍，还要单独训
2. **KL 塞在 reward 里**：$r_t = r_\varphi(q,o_{\le t}) - \beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，让 advantage 的计算被 KL 污染

---

## 2. GRPO 的三处改动

### 改动 ①：砍掉 value function，用 group baseline 估 advantage

对同一个 prompt $q$，**采样一组 $G$ 个输出** $\{o_1,\dots,o_G\}$，用 reward model（或规则）打分得 $\{r_1,\dots,r_G\}$。

**Outcome Supervision**（DeepSeekMath §4.1.2，sequence-level reward）的 advantage：

$$
\boxed{\;\hat{A}_i \;=\; \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}\;}
$$

- 这就是 **group-relative 的 z-score 归一化**。
- 比 group 均值好的输出 → 正 advantage；差的 → 负 advantage。
- **不需要 $V_\psi$**：baseline 用组内统计量代替 learned critic。省一个模型。

> **Process Supervision**（§4.1.3）：每个推理步打 step-level reward $\{r^i_1,\dots,r^i_K\}$，advantage 沿 token 广播 + 同样做 group 归一化。R1 用的是 outcome 为主。

### 改动 ②：KL 从 reward 里挪到 loss 里

PPO 把 KL 塞进 reward（$r_t$ 里减 KL），导致 advantage $\hat{A}_t$ 的计算被 KL 干扰。
GRPO **把 KL 直接作为 loss 的独立项**加在最后：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}}\bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_t \min\!\Big(\rho_{i,t}\hat{A}_{i,t},\;\text{clip}(\rho_{i,t},1\!-\!\varepsilon,1\!+\!\varepsilon)\hat{A}_{i,t}\Big) \;-\; \beta\,\mathbb{D}_{KL}[\pi_\theta\|\pi_{\text{ref}}]\bigg]
$$

这样 advantage $\hat{A}_{i,t}$ 保持"干净"（纯 reward 信号），KL 只在最后整体约束。

### 改动 ③：KL 用无偏估计量（Schulman 2020）

直接用 $\log(\pi_\theta/\pi_{\text{ref}})$ 估 KL 是有偏的（低估）。GRPO 用 **k3 估计量**（Schulman 2016/2020 blog）：

$$
\hat{\mathbb{D}}_{KL}[\pi_\theta\|\pi_{\text{ref}}] \;=\; \frac{\pi_{\text{ref}}(o_t|\cdot)}{\pi_\theta(o_t|\cdot)} - \log\frac{\pi_{\text{ref}}(o_t|\cdot)}{\pi_\theta(o_t|\cdot)} - 1
$$

无偏、方差更可控。这是 GRPO 训练稳定的一个细节工程点。

---

## 3. GRPO 完整流程（伪代码）

```
for 每个 prompt q:
    1. 从 π_θ_old 采样 G 个输出 {o_1,...,o_G}
    2. reward model / 规则 打分 → {r_1,...,r_G}
    3. group 归一化: Â_i = (r_i − mean) / std
    4. 广播到 token: Â_{i,t} = Â_i  (outcome supervision)
    5. 算 PPO clipped surrogate + KL(π_θ || π_ref)  → 更新 θ
```

**一句话**：sample 一组、组内排序当 advantage、PPO 更新、KL 拉回 reference。

---

## 4. 为什么 GRPO 适合 reasoning（R1 的选择）

| 理由 | 说明 |
|------|------|
| **奖励可验证** | 数学/代码的 reward 是规则判定的（答案对错），**没有 reward model，就没有 reward hacking 的入口** |
| **需要探索** | reasoning 要模型自己"想出来"长链，off-policy 的 DPO（只用固定偏好对）探索性不足；on-policy GRPO 能发现新路径 |
| **轻量** | 砍掉 critic，7B/671B 都能训得动 |

R1-Zero 就是**纯 GRPO on DeepSeek-V3-Base**（不先 SFT reasoning traces），结果 reasoning 能力自己涌现——这是 GRPO 探索性的最强证据。

---

## 5. GRPO vs DPO：三个本质差异

| 维度 | DPO | GRPO |
|------|-----|------|
| **① 采样范式** | **off-policy**：用固定偏好数据集 $(x,y_w,y_l)$，训练时不采样 | **on-policy**：每个 prompt 现采 $G$ 个输出打分 |
| **② reward 来源** | **隐式 reward**：$\beta\log(\pi_\theta/\pi_{\text{ref}})$，不需要 reward model | **显式 reward**：reward model 或规则（数学正确性），**需要 reward 信号** |
| **③ reference model 角色** | 在 loss 里以 **logprob 比值**形式出现（$\log\frac{\pi_\theta}{\pi_{\text{ref}}}$），隐式 KL | 在 loss 里以 **独立 KL 项**出现（采样时算 online KL），显式 KL |

> **共同点**：两者**都保留 reference model**做 KL 约束。区别在于 KL 是隐式（DPO，藏在 sigmoid 里）还是显式（GRPO，独立 loss 项）。
> **GRPO 真正砍掉的是 PPO 的 critic/value 网络**，不是 reference model。

---

## 6. Reward Hacking：是什么、怎么检测（Day 5 整理）

### 定义
模型找到了 reward function 的漏洞——**reward 分数持续上升，但实际输出质量下降/退化**。

### 常见形态
- **长度 hacking**：reward 偏好长答案 → 模型灌水
- **格式 hacking**：reward 看关键词 → 模型堆关键词不解决问题
- **捷径 hacking**：数学 reward 只看最终答案 → 模型背答案/乱写过程蒙对

### GRPO 里的检测
1. **holdout + 人工抽样**：定期在冻结的验证集上人工评分，reward↑ 但人工分↓ → hacking
2. **多 reward 交叉验证**：用 2+ 个独立 reward model，若分数分歧变大 → 某个被 hack
3. **监控 KL**：$\mathbb{D}_{KL}[\pi_\theta\|\pi_{\text{ref}}]$ 突然飙升 → 策略暴走，大概率在 hack
4. **R1 的解法**：**用规则 reward（数学/代码可验证）替代神经 reward model**，从源头消除 hacking——这是 R1 能纯 RL 训稳的关键。

---

## 7. DeepSeek-R1 的 GRPO 应用管线

```
DeepSeek-V3-Base (预训练)
        │
        ├─→ R1-Zero: 纯 GRPO (规则 reward: 数学正确性 + 格式)
        │           → reasoning 涌现 ("aha moment"), 但可读性差
        │
        └─→ R1 (多阶段, 工程优化):
              ① cold-start SFT (少量高质量 reasoning 样本, 改善格式)
              ② RL (GRPO, 推理能力)
              ③ rejection sampling (RL 模型生成 → 只留高质量 → 造 SFT 数据)
              ④ full SFT (用 ③ 的数据 + 通用数据再训)
              ⑤ final RL (GRPO, 同时优化推理 + 通用能力)
```

**关键洞察**：
- **R1-Zero 证明** reasoning 能从纯 RL（GRPO）涌现，不需要人类标注的 reasoning trace
- **R1 工程化**：cold-start 解决可读性，rejection sampling 把 RL 能力蒸馏回 SFT 数据（反向蒸馏），多轮迭代

---

## 8. 要点速答

**Q: GRPO 和 PPO 的核心区别？**
A: GRPO 砍掉了 value/critic 网络，用"一组采样的 reward 做组内 z-score 归一化"代替 learned baseline 估 advantage。另外把 KL 从 reward 里挪到 loss 外层，并用无偏 KL 估计量。

**Q: GRPO 和 DPO 在 reference model 上的差异？**
A: **两者都有 reference model**（常见误解是 GRPO 没有）。差异在角色：DPO 的 reference 以 logprob 比值隐式进 loss；GRPO 以独立 KL 项显式进 loss。GRPO 砍的是 critic，不是 reference。

**Q: 为什么 R1 选 GRPO 而不是 DPO？**
A: reasoning 需要模型自己探索出长推理链（on-policy），DPO 用固定偏好对、探索性不足；且数学/代码 reward 可用规则验证，没有 reward model 就没 hacking 入口，正好发挥 GRPO 的优势。

**Q: reward hacking 怎么检测？**
A: 核心：reward↑ 但 holdout 人工分↓ = hacking。辅助：多 reward 交叉验证、监控 KL 是否飙升。根治：用可验证的规则 reward 替代神经 reward model（R1 做法）。

---

> 配套 [derivation_dpo_detailed.md](derivation_dpo_detailed.md)（DPO 推导）和 [../notes/week13_dpo_grpo_comparison.md](../notes/week13_dpo_grpo_comparison.md)（5 篇论文笔记 + 方法选型表）。
