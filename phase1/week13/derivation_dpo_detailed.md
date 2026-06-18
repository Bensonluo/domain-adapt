# DPO Loss 完整推导：从 RLHF 到闭式解

> **一句话**：DPO 用一个数学技巧（闭式解 + Bradley-Terry）把"RLHF 的强化学习"变成"一个分类 loss"，
> 既不需要训练 reward model，也不需要在线采样。这篇是手推全过程，每一步都讲清**为什么**。
>
> 配套论文：Rafailov et al. 2023, *Direct Preference Optimization* (arXiv:2305.18290)

---

## 0. 起点：RLHF 到底在解什么问题

RLHF（Ouyang et al. 2022）的目标是：**在奖励最大化 和 不偏离参考模型之间找平衡**。

$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi_\theta(\cdot|x)}\big[r(x,y)\big] \;-\; \beta\, \mathbb{D}_{KL}\!\Big[\pi_\theta(\cdot|x)\,\big\|\,\pi_{\text{ref}}(\cdot|x)\Big]
$$

- $\pi_\theta$：要训练的策略（LM）
- $r(x,y)$：reward 函数（RLHF 里要先训一个 reward model 去近似它）
- $\pi_{\text{ref}}$：参考模型（通常是 SFT 后的模型，提供"别走太远"的锚点）
- $\beta$：KL 约束强度（$\beta\!\to\!\infty$ 完全不动；$\beta\!\to\!0$ 只要 reward 不顾一切）

**RLHF 的工程痛点**（DPO 要消灭的）：
1. 要单独训一个 reward model（两阶段，误差累积）
2. 训练时要**在线采样** $y\sim\pi_\theta(\cdot|x)$ 算 reward（慢、不稳）
3. PPO 还要训一个 value/critic network（显存翻倍）

DPO 的野心：**能不能不要 reward model、不要 RL 采样，直接拿偏好数据训 LM？**

---

## 1. 关键一步：RLHF 目标有闭式最优解

把上面的目标对**单个** $x$ 写开（$\pi_\theta(\cdot|x)$ 是分布，约束是归一化 $\sum_y \pi(y|x)=1$），用 Lagrangian 求解，得到**最优策略的解析形式**：

$$
\boxed{\;\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\Big(\frac{1}{\beta}r(x,y)\Big)\;}
$$

其中 **配分函数**（归一化常数）：

$$
Z(x) \;=\; \sum_{y}\pi_{\text{ref}}(y|x)\,\exp\!\Big(\frac{1}{\beta}r(x,y)\Big)
$$

**直觉**：最优策略 = 参考策略乘一个"reward 的玻尔兹曼权重"，再归一化。reward 高的 $y$ 概率被放大，低的被压低，但永远保留 $\pi_{\text{ref}}$ 的底色（KL 约束的体现）。

**这个闭式解本身不可用**：$Z(x)$ 要对所有 $y$ 求和（$y$ 是任意长文本，无穷项），intractable。这正是 RLHF 当初要用 RL 而不是直接解这个式子的原因。

---

## 2. 反解：把 reward 用策略表示出来（implicit reward）

闭式解把 $\pi^*$ 表成了 $r$ 的函数。**反过来**，把 $r$ 表成 $\pi^*$ 的函数：

从 $\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$ 出发，两边除以 $\pi_{\text{ref}}$ 再取对数：

$$
\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \;=\; \frac{1}{\beta}r(x,y) - \log Z(x)
$$

$$
\boxed{\;r(x,y) \;=\; \beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \;+\; \beta\log Z(x)\;}
$$

**这是 DPO 的核心 reparameterization**：任意一个策略 $\pi^*$ 都"隐含"定义了一个 reward 函数。
也就是说——**reward model 是多余的**，语言模型自己的 logprob 比值就是 reward。

> 注意 $Z(x)$ 还在式子里。下一节看它怎么被消掉。

---

## 3. 代入 Bradley-Terry：Z(x) 神奇抵消

人类偏好数据形如 $(x, y_w, y_l)$——$y_w$ 是 chosen，$y_l$ 是 rejected。
**Bradley-Terry 模型**假设：偏好的概率 = reward 差的 sigmoid：

$$
p(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r(x,y_w) - r(x,y_l)\big), \qquad \sigma(z)=\frac{1}{1+e^{-z}}
$$

把第 2 节的 $r(x,y)$ 代入：

$$
r(x,y_w) - r(x,y_l) \;=\; \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \cancel{\beta\log Z(x)} \;-\; \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \cancel{\beta\log Z(x)}
$$

**$Z(x)$ 在 chosen 和 rejected 之间完全抵消！** 这就是 DPO 能绕开 intractable 配分函数的根本原因。

$$
p(y_w \succ y_l \mid x) \;=\; \sigma\!\Bigg(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \;-\; \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigg)
$$

---

## 4. DPO Loss：一个二分类交叉熵

对偏好数据集做最大似然，取负对数，得到 **DPO loss**：

$$
\boxed{\;\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\bigg[\log\sigma\!\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)\bigg]\;}
$$

**这就完事了。** 整个 loss 只需要：
- 当前策略 $\pi_\theta$ 对 $y_w, y_l$ 的 logprob
- 参考策略 $\pi_{\text{ref}}$ 对 $y_w, y_l$ 的 logprob（reference 冻住，前向一次即可）

**没有 reward model，没有在线采样，没有 critic network。** 一个标准的二分类交叉熵，能直接用偏好数据训。

---

## 5. 三个关键洞察（面试常问）

### (a) 为什么 $Z(x)$ 能抵消？——Bradley-Terry 的功劳

$Z(x)$ 只依赖 $x$（不依赖 $y$）。BT 模型算的是 $r(x,y_w)-r(x,y_l)$，**两个 $y$ 共享同一个 $x$**，所以 $\beta\log Z(x)$ 一加一减正好消掉。

> 反过来说：如果偏好模型不是"差"的形式（比如 KTO 用的是单边的效用函数），$Z(x)$ 就不会这么干净地消掉——KTO 因此走了另一条路（prospect theory）。

### (b) $\beta$ 在控制什么？

$\beta$ 是 KL 约束强度的倒数（公式里是 $1/\beta$ 在 exponent）：

| $\beta$ | 行为 | 风险 |
|---------|------|------|
| 大（如 0.5） | 紧贴 $\pi_{\text{ref}}$，温和对齐 | 对齐不足 |
| 小（如 0.01） | 激进远离 $\pi_{\text{ref}}$ | 过拟合、reward hacking、退化 |
| 典型值 | 0.1 | 平衡 |

### (c) Reference model 为什么必须冻住？

$\pi_{\text{ref}}$ 是 KL 锚点。如果它也跟着训，"距离"就没有参照系了，loss 会无界增长（这正是 IPO 批评 DPO 的点）。实践中 $\pi_{\text{ref}}$ = SFT 后的模型副本，参数 freeze，只做前向算 logprob。

---

## 6. DPO 的梯度直觉（为什么它能 work）

对 $\log\pi_\theta(y_w|x)$ 求梯度（简化符号 $\hat{r}=\beta\log(\pi_\theta/\pi_{\text{ref}})$）：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} \;=\; -\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat{r}_l-\hat{r}_w)}_{\text{权重 } \in[0,1]}\,\big(\nabla_\theta\log\pi_\theta(y_w|x) - \nabla_\theta\log\pi_\theta(y_l|x)\big)\Big]
$$

- $\sigma(\hat{r}_l-\hat{r}_w)$：**模型当前已经分对 chosen/rejected 的程度**。
- 已经分得很开（$\hat{r}_w\gg\hat{r}_l$）→ 权重趋近 0 → **不再用力**（自适应停止）。
- 还没分开 → 权重大 → 继续推高 $y_w$、压低 $y_l$。

这解释了 DPO 的"自我调节"：越简单的偏好越早停止学习，难偏好继续用力。

---

## 7. DPO 的 Failure Modes（Day 5 整理点）

| 失败模式 | 表现 | 根因 | 缓解 |
|----------|------|------|------|
| **长度偏差** | 输出越来越长，质量没涨 | 长 $y$ 的 logprob 机制偏置 | length-regularized DPO / SimPO |
| **偏好噪声敏感** | 标注不一致就崩 | BT 假设偏好是确定的 | IPO（加正则）、数据清洗 |
| **分布偏移** | 训久后 $\pi_\theta$ 偏离数据集分布 | off-policy，无在线采样 | 迭代 DPO / online DPO |
| **Objective 无界** | logprob ratio 推到无穷 | DPO loss 无上界 → 过拟合 | IPO（显式正则项） |

---

## 8. 面试速答卡

**Q: DPO 为什么不需要 reward model？**
A: 因为 RLHF 目标有闭式最优解 $\pi^*=\frac{1}{Z}\pi_{\text{ref}}\exp(r/\beta)$，反解出 $r=\beta\log(\pi^*/\pi_{\text{ref}})+\beta\log Z$——reward 被 reparameterize 成了策略的 logprob 比。语言模型本身就是 reward model。

**Q: 闭式解里 $Z(x)$ 不可算，DPO 怎么处理的？**
A: 代入 Bradley-Terry 偏好模型时，算的是 $r(x,y_w)-r(x,y_l)$，两个 $y$ 共享同一个 $x$，$\beta\log Z(x)$ 一加一减正好抵消。这是 DPO 的数学核心。

**Q: DPO 和 RLHF 优化的是同一个目标吗？**
A: 是。DPO 是 KL 约束 reward 最大化那个目标的**精确转化**（在 BT 假设下），不是近似。只是把求解方式从 RL 换成了监督学习。

**Q: DPO 最大的弱点？**
A: off-policy + objective 无界。数据分布外的偏好学不好，长训会过拟合（logprob ratio 发散）→ IPO/KTO 就是为修这个。

---

## 附：推导链路一图流

```
RLHF 目标 (KL 约束 reward 最大化)
        │  ① 求闭式最优解 (Lagrangian)
        ▼
π*(y|x) = (1/Z(x)) · π_ref · exp(r/β)        ← Z(x) intractable, 本来用不了
        │  ② 反解 reward (reparameterization)
        ▼
r(x,y) = β·log(π*/π_ref) + β·log Z(x)        ← reward 变成 logprob 比
        │  ③ 代入 Bradley-Terry 偏好模型
        ▼
p(yw≻yl) = σ(r_w − r_l)  →  Z(x) 抵消       ← 同一个 x, Z 一加一减消掉
        │  ④ 负对数似然
        ▼
L_DPO = −E[ log σ( β·log(πθ(yw)/πref(yw)) − β·log(πθ(yl)/πref(yl)) ) ]
                                             ← 纯监督 loss, 无 RM / 无 RL / 无 critic
```

> 下一篇 [derivation_grpo.md](derivation_grpo.md)：GRPO 走的是另一条路——保留 RL 在线采样，但砍掉 critic，用 group baseline 估 advantage。
