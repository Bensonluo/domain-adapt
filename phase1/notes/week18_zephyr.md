# Zephyr-7B 论文精读笔记

> 论文: ZEPHYR: Direct Distillation of LM Alignment (Tunstall et al., Hugging Face, 2023)
> 链接: https://arxiv.org/abs/2310.16944
> Student: Mistral-7B；Teacher: GPT-3.5 / GPT-4（仅 API）
> PDF: [../week18/papers/Zephyr_2310.16944.pdf](../week18/papers/Zephyr_2310.16944.pdf)

---

## 核心观点 (TL;DR)

```
用纯蒸馏（distillation）把大模型的对齐能力（alignment）搬到 7B 小模型。
不需要任何人工标注，不需要 RL 的在线采样。

三步管线（复刻 InstructGPT 的 SFT → RM → RL，但全换成蒸馏版）:

  Step 1  dSFT  蒸馏式监督微调
          用 UltraChat（GPT-3.5 自举造的多轮对话，self-instruct 风格）
          → 让 Mistral 学会"对话格式 + 基本能力"

  Step 2  AIF   AI 反馈收集
          对每个 prompt，4 个模型各生成一个回答，GPT-4 打分
          → 取最高分当 y_w（chosen），随机取一个低分当 y_l（rejected）
          → 得到 (prompt, chosen, rejected) 偏好对（无人工！）

  Step 3  dDPO  蒸馏式直接偏好优化
          用 Step 1 的 dSFT 模型当 reference，在偏好对上跑 DPO
          → 不用 PPO 在线采样，离线就能优化偏好

结果: Zephyr-7B 在 MT-Bench 上 7.34，超过 Llama2-Chat-70B（6.86）！
      一个 7B 蒸馏模型，干翻 70B RLHF 模型。

一句话：蒸馏不仅能传"能力"(SFT)，还能传"对齐"(DPO)。
      把 InstructGPT 的三阶段（SFT+RM+PPO）全替换成蒸馏版（dSFT+AIF+dDPO），
      是"用大模型 API 对齐小模型"的完整范式。
```

---

## 核心问题

之前的蒸馏工作（Alpaca、Vicuna）只蒸了 **SFT 阶段**——让小模型学会"回答问题"。但这些模型不够"intent aligned"（意图对齐）：回答可能冗长、不直接、不符合人类偏好。

**对齐**这步，传统上靠 RLHF（InstructGPT、Llama2-Chat），需要：人工偏好标注 + 训练 reward model + PPO 在线采样。贵、慢、工程复杂。

**Zephyr 的问题**：能不能**不用人工标注、不用 PPO**，纯靠大模型的 AI 反馈（AIF）+ DPO，把对齐也蒸馏下来？

---

## 方法：dSFT → AIF → dDPO

```
┌──────────────────────────────────────────────────────────────┐
│ 基座: Mistral-7B (当时最强 7B 开源基座)                         │
└──────────────────────────────────────────────────────────────┘
                              │
   ┌──────────────────────────┴──────────────────────────┐
   │  Step 1: dSFT (distilled SFT)                       │
   │  ─────────────────────────────                       │
   │  数据: UltraChat (1.47M 多轮对话, GPT-3.5 自举造的)    │
   │  问题: 原始数据有 ~5% 大小写错误 + "我没有个人经历"     │
   │        这类 AI 味回答 → 用 truecasing + 过滤清洗        │
   │        → 得到 ~200K 干净样本                          │
   │  训练: SFT, lr=2e-5, cosine, batch 512, seq 2048     │
   │  结果: π_dSFT —— 会对话, 但不够 aligned              │
   └──────────────────────────┬──────────────────────────┘
                              │
   ┌──────────────────────────┴──────────────────────────┐
   │  Step 2: AIF (AI Feedback)                          │
   │  ─────────────────────────────                       │
   │  数据源: UltraFeedback (64K prompts)                 │
   │  流程(离线, 不采样 student):                          │
   │    每个 prompt x → 喂给 4 个模型                       │
   │      (Claude / Falcon / Llama / ... 各生成 y1..y4)   │
   │    → GPT-4 给每个回答打分 s1..s4                      │
   │    → y_w = 最高分; y_l = 随机选一个低分                │
   │      (不选最低分, 是为了让 DPO 任务更难/更多样)        │
   │  结果: 64K 偏好三元组 (x, y_w, y_l), 零人工标注       │
   └──────────────────────────┬──────────────────────────┘
                              │
   ┌──────────────────────────┴──────────────────────────┐
   │  Step 3: dDPO (distilled DPO)                       │
   │  ─────────────────────────────                       │
   │  目标: 让 π_θ 更偏好 y_w 而非 y_l                     │
   │  关键: reference = π_dSFT (不是原 Mistral)            │
   │  loss (就是标准 DPO, 见 week13):                      │
   │    max E[ log σ( β log π(y_w|x)/π_dSFT(y_w|x)       │
   │               - β log π(y_l|x)/π_dSFT(y_l|x) ) ]     │
   │  超参: lr=5e-7, linear, β=0.1, batch 32              │
   │  训练: 1 epoch SFT + 3 epochs DPO (消融最优)          │
   │  结果: Zephyr-7B —— aligned 的 7B chat 模型          │
   └──────────────────────────────────────────────────────┘

成本: 16 × A100, 每个阶段 2-4 小时。总训练成本 << RLHF。
```

**三个关键设计决策**：

1. **y_l 随机选，不选最低分**：选最低分会让偏好对比太容易（最好 vs 最差），DPO 学不到细粒度；随机选让任务更难，逼模型学"为什么这个比那个好"。
2. **reference 是 dSFT 模型**：DPO 需要 reference 来约束 KL。用 dSFT 版而非 raw Mistral，保证 reference 已经"会对话"，DPO 只是在此基础上调偏好。
3. **β=0.1**：控制偏离 reference 的强度（week13 概念）。这里偏保守。

---

## 关键实验与数字

### 主结果：Chat benchmark（Table 1）

| 模型 | 参数量 | 对齐方式 | MT-Bench | AlpacaEval |
|------|-------|---------|----------|------------|
| MPT-Chat | 7B | dSFT | 5.42 | - |
| Mistral-Instruct | 7B | - | 6.84 | - |
| Xwin-LM | 7B | dPPO | 6.19 | 87.83 |
| **Zephyr** | **7B** | **dDPO** | **7.34** | **90.60** |
| Guanaco | 65B | SFT | 6.41 | 71.80 |
| Vicuna | 33B | dSFT | 7.12 | 88.99 |
| **Llama2-Chat** | **70B** | **RLHF** | **6.86** | **92.66** |
| GPT-3.5-turbo | - | RLHF | 7.94 | 89.37 |
| Claude 2 | - | RLHF | 8.06 | 91.36 |
| GPT-4 | - | RLHF | 8.99 | 95.28 |

- **7B 里 SOTA**：Zephyr 7.34，碾压所有 7B 对手（含 dPPO 的 Xwin）
- **超过 Llama2-Chat-70B**（6.86）：一个 7B 蒸馏模型，在 MT-Bench 上干翻 70B + RLHF + 大量人工反馈的模型！这是当时最震撼的结果。
- 但 AlpacaEval 上仍略输 Llama2-Chat-70B（90.6 vs 92.66），且数学/编码弱（见 Figure 1 分项）

### Ablation：每一步都有用吗？（Table 3）

| 配置 | MT-Bench | AlpacaEval |
|------|----------|------------|
| dDPO - dSFT（直接对 raw 模型 DPO） | 4.76 | 30.76 |
| dSFT-1（只 SFT） | 6.64 | 85.65 |
| dSFT-2（SFT + 在偏好数据上再做 SFT） | 6.19 | 78.54 |
| **dDPO + dSFT（完整 Zephyr）** | **7.00** | **86.07** |

**三个关键发现**：

1. **SFT 是 DPO 的前提**。跳过 SFT 直接 DPO → MT-Bench 暴跌到 4.76。原因（附录 A.2）：模型连 chat template（`<|user|>` `<|assistant|>`）都没学会，DPO 在乱码格式上优化。这呼应 week13 的结论——**DPO 要在 SFT 之后**。

2. **dSFT-2（在偏好数据上再 SFT 最高分答案）反而更差**（6.19 < 6.64）。这说明：**偏好信号 > 单纯模仿好答案**。直接学 y_w 的输出（SFT）不如学"w 比 l 好"的相对关系（DPO）。

3. **dDPO + dSFT 涨点显著**（6.64 → 7.00）。证明 DPO 这一步独立有效。

### 过拟合但不掉点（Figure 3）

```
观察: DPO 训练 1 epoch 后，训练集准确率 → 100%（严重过拟合）
预期: 过拟合应该掉下游分

实际: 1 epoch SFT + 3 epochs DPO 反而最优！
BUT: 如果 SFT 训超过 1 epoch，DPO 训久了反而掉分

解读:
  - DPO 的"过拟合"（训练集分对）不代表下游退化
    (训练集 y_w vs y_l 区分明显，分对很容易，不代表泛化差)
  - SFT 过拟合才危险 (SFT 多 epoch → 模型死记训练数据 → DPO 失去调整空间)
  → 经验: SFT 少训(1 epoch), DPO 可以多训(3 epochs)
```

---

## 为什么有效？（蒸馏对齐的本质）

```
传统 RLHF 三阶段:           Zephyr 的蒸馏版:
  SFT (人工指令)        →    dSFT (GPT-3.5 造的指令)
  RM (人工偏好)         →    AIF (GPT-4 打分当偏好)
  PPO (在线采样优化)    →    dDPO (离线 DPO)

为什么蒸馏版能逼近 RLHF 版?

1. 对齐的"信号"不需要来自人类
   GPT-4 打分 ≈ 聚合的人类偏好 (GPT-4 本身被 RLHF 对齐过)
   → 用 GPT-4 当裁判 = 借用 OpenAI 已投入的海量人工偏好
   → 这是"二手 RLHF"，便宜但有效

2. DPO 绕开了 PPO 的复杂度
   PPO: 训 RM + 在线采样 + critic + KL 惩罚 (week13)
   DPO: 离线偏好对 + 一个 loss (week13)
   → 离线、稳定、便宜

3. Mistral-7B 基座够强
   基座已有 latent capability, 蒸馏只是"激活 + 对齐"
   (和 Self-Instruct 的逻辑一致: 能力是潜在的)

本质: Zephyr = "用强模型的 API，把对齐能力批发给小模型"
      teacher (GPT-4) 不需要给权重，只需要给"判断"(打分)
      → 这是 response/feedback distillation 的极致
```

---

## 局限性（论文承认 + 我补充）

```
1. 评测偏见
   MT-Bench/AlpacaEval 都用 GPT-4 当裁判
   → GPT-4 偏好"像自己/冗长"的回答 (论文自己点名)
   → Zephyr 部分受益于这种偏见 (因为它的偏好数据也是 GPT-4 打的)
   → 胜过 Llama2-70B 的结论要打折扣

2. 安全性没做
   论文明说: 只优化 helpfulness, 没碰 safety
   (合成有害数据难造, 是 open problem)
   → Zephyr 可能更容易被越狱

3. 数学/编码弱 (Figure 1)
   这些任务需要"可验证的正确性", 蒸馏只能传"风格"传不了"能力"
   → 这正是 GRPO/规则 reward 的主场 (week13 R1 路线)

4. 天花板 = teacher
   Zephyr 的上限是 GPT-4 的判断质量
   (Gudibande 2023 "The false promise of imitating proprietary LLMs"
    证明蒸馏模型无法超越 teacher, 且会继承 teacher 的盲点)
```

---

## 在蒸馏谱系中的位置（串起三篇）

```
DistilBERT (2019)        Self-Instruct (2022)        Zephyr (2023)
────────────────         ──────────────────          ─────────────
"怎么蒸"                  "数据从哪来"                "蒸什么 + 全流程"
Feature distillation     合成数据自举                对齐蒸馏 (dSFT+dDPO)
有 teacher 权重          只有 API                    只有 API
encoder 分类             decoder 对话                decoder chat 对齐
学 hidden + soft label   学 output 文本              学 output + 偏好

DistilBERT 回答: 蒸馏的技术细节 (loss 怎么设计)
Self-Instruct 回答: 蒸馏的数据怎么自动造
Zephyr 回答: 把 SFT + 对齐 两阶段都蒸馏化, 端到端

三者合起来 = 一套完整的"用大模型造小模型"的方法论:
  技术底座 (DistilBERT) + 数据方法 (Self-Instruct) + 对齐落地 (Zephyr)
```

---

## 与 week13（DPO/GRPO）的衔接

```
Zephyr 用的是 DPO (week13 的核心论文之一)。
对照 week13 的 DPO loss:

  L_DPO = -E[ log σ( β log π(y_w|x)/π_ref(y_w|x)
                  - β log π(y_l|x)/π_ref(y_l|x) ) ]

Zephyr 的特殊性:
  - π_ref = π_dSFT (不是原始基座, 是 SFT 后的模型)
  - 偏好数据来自 GPT-4 打分 (AIF), 不是人工
  - offline (不采样), 这正是 DPO 相对 PPO 的优势

和 GRPO 路线 (R1) 的对比:
  Zephyr (dDPO): offline, 偏好驱动, 传"对齐风格"
  R1 (GRPO):    online, 规则 reward 驱动, 探索"推理能力"
  → Zephyr 适合"对话/写作", R1 适合"数学/代码"
  → 我的垂域方向: 对齐用 DPO/Zephyr 范式, reasoning 用 GRPO 范式
```

---

## 我的 takeaway

1. **Zephyr 是"对齐蒸馏"的里程碑**。它证明了：对齐（alignment）这个原本只有 RLHF 能做的事，可以纯靠大模型 API + DPO 完成，不需要人工标注。这把开源模型的能力门槛大幅拉低。
2. **核心洞察：对齐信号可以"二手获取"**。GPT-4 当裁判 = 复用 OpenAI 投入的海量 RLHF 偏好。用 teacher 的"判断"代替人工"判断"，是 distillation 在对齐层面的体现。
3. **dSFT-2 ablation 是最反直觉的发现**：直接 SFT 好答案（学 y_w 输出）反而不如 DPO（学 w>l 的相对关系）。这印证了 week13 的观点——**偏好学习的"相对性"比模仿学习的"绝对性"更高效**。
4. **SFT 是 DPO 的地基，不可省**。跳过 SFT 直接 DPO，模型连 chat template 都学不会。这和 week13 的结论完全一致：**先 SFT 再 DPO/GRPO** 是对齐管线的铁律。
5. **对我的垂域项目的直接启示**：如果要对齐垂域 7B 模型，Zephyr 范式（领域 dSFT + GPT-4 打分造领域偏好 + dDPO）比从头搞 RLHF 现实得多。这条路线我会用到后续 week。
