# Week 18：三种蒸馏方法对比 + 3 篇论文串联

> 把 3 篇论文（DistilBERT / Self-Instruct / Zephyr）串成**蒸馏方法的全景图**，
> 给出"什么场景该用哪种蒸馏"的决策框架，并解答 README 的自测题。
>
> 配套精读笔记：[week18_distilbert.md](week18_distilbert.md) · [week18_self_instruct.md](week18_self_instruct.md) · [week18_zephyr.md](week18_zephyr.md)
> PDF 原文：[../week18/papers/](../week18/papers/)

---

## 一、全景：三种蒸馏方法的演化树

```
            Hinton 2015 (知识蒸馏奠基)
            "soft label 携带 dark knowledge"
                      │
          ┌───────────┼───────────────────────┐
          ▼           ▼                       ▼
     Response      Feature                 On-Policy
     蒸馏          蒸馏                    蒸馏
   "学输出"       "学内部表示"            "学策略"
      │              │                       │
      │         DistilBERT (2019)            │
      │         triple loss: L_ce            │
      │         + L_mlm + L_cos              │
      │         (response+feature 混合)       │
      │              │                       │
      ├──────────────┘                       │
      │                                      │
   Self-Instruct (2022)              R1 rejection sampling (2025)
   "学输出文本"                      "student 生成 → teacher/reward
   合成数据自举                       打分 → 固化进 SFT"
   (response 的数据层)                (on-policy, 见 week13)
      │
   Zephyr (2023)
   "学输出 + 学偏好"
   dSFT (response) + dDPO (偏好)
   (response 的对齐层)


关键脉络:
  - Feature (DistilBERT): 有 teacher 权重 → 学深层表示
  - Response (Self-Instruct/Zephyr): 只有 teacher API → 学输出
  - On-policy (R1): student 自己生成 → 学策略
  三者从"浅"到"深"，从"便宜"到"贵"
```

---

## 二、三种蒸馏方法详解

### 1. Response Distillation（响应蒸馏）

```
做法: 大模型生成输出 → 小模型当 ground truth 学
      (本质上是用 teacher 的输出做 SFT)

代表: Self-Instruct (造数据) / Alpaca / Vicuna / Zephyr 的 dSFT 阶段

学什么: teacher 的"输出文本" (只学成品)
        student 看不到 teacher 的内部状态、不确定性、推理过程

成本: 低 —— 只需调 teacher 的 API (按 token 付费)
深度: 浅 —— 只模仿最终输出

优点:
  + 不需要 teacher 权重 (黑盒 API 就行)
  + 实现简单 (就是 SFT)
  + 可以用闭源最强模型 (GPT-4) 当 teacher

缺点:
  - 学不到 teacher 的内部表示
  - 天花板 = teacher 的输出质量
  - 无法超越 teacher (Gudibande 2023 已证明)
```

### 2. Feature Distillation（特征蒸馏）

```
做法: 对齐 teacher 和 student 的中间层表示 (hidden states)
      student 不仅学输出，还学"内部特征"

代表: DistilBERT (triple loss 的 L_cos)
      后续: TinyBERT (加更多层对齐) / MiniLM / attention 蒸馏

学什么: teacher 的"中间层激活方向"
        (不只学结论，学得出结论的中间过程)

成本: 中 —— 需要访问 teacher 的权重/hidden states (白盒)
深度: 中 —— 学内部表示

为什么需要 projection layer (自测题 2 的核心):
  teacher 和 student 维度往往不同 (如 teacher 768d, student 384d)
  → 没法直接算 hidden state 的距离
  → 加一个可学习的线性层 W: student_dim → teacher_dim
  → 对齐的是 W·h_student 和 h_teacher
  DistilBERT 不需要 projection, 因为它故意保持 hidden size 不变 (都 768)
  → 这是它"只砍层数不砍宽度"设计的一个副产品

优点:
  + 学到的表示更深 (内部结构对齐)
  + 相同参数下，比 response 蒸馏效果好
  + 不依赖 teacher 的输出质量 (学的是表示)

缺点:
  - 需要 teacher 权重 (白盒) —— 闭源模型用不了
  - 工程复杂 (要对齐多层 + 加 projection)
```

### 3. On-Policy Distillation（在线蒸馏）

```
做法: student 自己生成输出 → teacher/reward 打分 → student 据此改进
      (student 在自己的分布上学习，而非学 teacher 的固定输出)

代表: R1 的 rejection sampling (week13)
      student 采样多条推理链 → 规则判断对错 → 对的当 SFT 数据 → 再训
      也包括 online DPO / RL 蒸馏

学什么: student 自己探索出的"好策略"
        (不是抄 teacher，而是在 reward 引导下自己找)

成本: 高 —— 多轮交互 (生成 + 打分 + 重训，迭代)
深度: 深 —— 学策略 (policy-level)

优点:
  + 效果最好 (能探索 teacher 没示范的路径)
  + 可以超越 teacher (R1 在数学上超越 GPT-4 示范)
  + 适合 reasoning (探索性强)

缺点:
  - 最贵 (多轮迭代 + 大量采样)
  - 工程最复杂 (要在线生成 pipeline)
  - 需要 reward (规则 reward 最稳，神经 RM 有 hacking 风险)
```

---

## 三、对比大表（核心交付物）

| 维度 | **Response** | **Feature** | **On-Policy** |
|------|-------------|-------------|---------------|
| **做法** | teacher 输出 → student SFT | 对齐中间层 hidden state | student 生成 → 打分 → 改进 |
| **代表论文** | Self-Instruct, Zephyr(dSFT) | **DistilBERT** | R1 rejection sampling |
| **学什么** | 输出文本（成品） | 内部表示（过程） | 策略（探索） |
| **需要 teacher 权重?** | ❌ 只需 API | ✅ 需白盒 hidden | △ 需 reward 函数 |
| **成本** | 低（API 费） | 中（需访问权重） | 高（多轮迭代） |
| **深度** | 浅 | 中 | 深 |
| **能超越 teacher?** | ❌ 不能 | ❌ 不能 | ✅ 能（探索） |
| **典型 loss** | 交叉熵 (SFT) | MSE/cosine on hidden | DPO/PPO + reward |
| **适用阶段** | SFT / 对齐 | 预训练压缩 | RL / reasoning |
| **适合任务** | 对话/写作/通用 | 模型压缩部署 | 数学/代码/reasoning |
| **垂域应用** | 造领域指令数据 | 压缩领域大模型 | 领域推理任务 |

---

## 四、决策框架：什么场景用哪种

```
你的蒸馏目标是什么？
│
├─ 「我要把大模型压缩成可部署的小模型」
│   └─ 你有 teacher 权重吗?
│       ├─ 有 → Feature distillation (DistilBERT 范式)
│       │       triple loss: response + MLM + feature
│       └─ 没有(只有 API) → Response (SFT 在 teacher 输出上)
│
├─ 「我要让小模型学会对话/指令跟随」
│   └─ Response distillation
│       数据层: Self-Instruct 自举造指令
│       训练层: SFT on teacher 输出 (Alpaca/Vicuna 路线)
│
├─ 「我要让小模型对齐人类偏好」
│   └─ Response + 偏好蒸馏 (Zephyr 范式)
│       dSFT (Self-Instruct 造数据) → AIF (GPT-4 打分) → dDPO
│       不需人工标注, 不需 PPO
│
└─ 「我要让模型获得 reasoning 能力(数学/代码)」
    └─ On-policy distillation (R1 范式)
        student 采样 → 规则 reward → rejection sampling → SFT
        能超越 teacher, 但最贵
```

**实战速记**（对应我的垂域项目）：
- **压缩部署** → Feature（有权重）或 Response（只有 API）
- **领域对话** → Self-Instruct 造领域数据 + SFT
- **领域对齐** → Zephyr 范式（领域 dSFT + AIF + dDPO）
- **领域推理** → R1 范式（GRPO + 规则 reward）—— 见 week13

---

## 五、3 篇论文一句话串讲（知识脉络）

```
1. DistilBERT (2019) — 蒸馏的"技术底座"
   回答"怎么蒸": triple loss = response(L_ce) + 自身(L_mlm) + feature(L_cos)
   证明 feature distillation 能保住 97% 性能, 砍 40% 参数。
   前提: 有 teacher 权重 (白盒)。

2. Self-Instruct (2022) — 蒸馏的"数据方法"
   回答"数据从哪来": 让 LM 自举造指令数据 (175 种子 → 52K 指令)
   把人工标注从必需品变可选项。
   催生 Alpaca/Vicuna 整条"用 GPT 蒸馏开源模型"的路线。

3. Zephyr (2023) — 蒸馏的"对齐落地"
   回答"蒸什么 + 全流程": 把 SFT + 对齐 都蒸馏化
   dSFT (Self-Instruct 造数据) + AIF (GPT-4 打分) + dDPO
   7B 蒸馏模型干翻 70B RLHF 模型, 证明"对齐可二手获取"。

三者递进:
  技术 (DistilBERT) → 数据 (Self-Instruct) → 对齐 (Zephyr)
  从"有 teacher 权重"到"只有 API"的范式迁移
  从"压缩模型"到"传承对齐"的目标升级
```

---

## 六、README 自测题解答

### Q1: Response distillation 和普通 SFT 有什么区别？

```
表面看: 两者都是"拿 (输入, 输出) 对做监督学习", loss 都是交叉熵。
        从训练算法角度，几乎没区别。

核心区别在「数据来源 + 学到什么」:

  普通 SFT:
    - 数据来源: 人工标注 / 已有数据集 (如 SUPERNI)
    - 学的是: 人类写的"标准答案"
    - 天花板: 人工标注质量 + 数量 (贵、有限)

  Response distillation:
    - 数据来源: teacher 模型生成 (Self-Instruct 自举 / teacher 直接答)
    - 学的是: teacher 的"输出分布"(不只 hard label, 还可拿 soft label)
    - 天花板: teacher 的能力 (但数据可以无限造、便宜)

一句话:
  Response distillation = 用 teacher 输出替代人工标注的 SFT。
  训练流程一样, 但数据从"人工"换成"模型生成",
  于是成本骤降、规模可放大、多样性可控。

进阶区别 (带 soft label 时):
  纯 SFT 学 hard label (one-hot), 1 bit 信息;
  Response distillation 可以学 teacher 的 soft label (logits 分布),
  携带"类间关系"的 dark knowledge, 比人工标注信息量大。
  (这是 DistilBERT L_ce 和普通 SFT 的本质差异)
```

### Q2: Feature distillation 为什么要加 projection layer？

```
问题根源: teacher 和 student 的 hidden state 维度通常不同。

  例: teacher (BERT-base) hidden = 768d
      student (更小的模型) hidden = 384d

  要对齐它们 (算距离/相似度), 必须先映射到同一空间。

不加 projection 会怎样?
  → 维度不匹配, 根本没法算 loss (768 vs 384)
  → 或者强行要求 student 维度 = teacher 维度
    (但这限制了 student 的压缩空间, 不能砍宽度)

projection layer 的作用:
  加一个可学习的线性层 W: R^student_dim → R^teacher_dim

    对齐目标: W · h_student  ≈  h_teacher
              (projected)      (teacher)

  W 随 student 一起训练, 学一个"最佳映射"。
  对齐的是"投影后的 student 表示"和"teacher 表示"的方向/距离。

为什么用投影而非强制等维:
  - 解耦 student 架构和 teacher 架构 (student 可以任意窄)
  - 投影层参数少, 开销小
  - 投影本身可学, 自动找到两个空间的最优对齐

DistilBERT 的特例 (为什么它不用 projection):
  DistilBERT 故意保持 hidden size = 768 (和 BERT 相同), 只砍层数
  → student/teacher 维度天然一致 → 直接算 cosine, 不需投影
  → 这是它"只减层数不减宽度"设计决策的一个副产品

后续工作 (TinyBERT 等):
  要同时砍层数 + 砍宽度 → 维度不同 → 必须加 projection
```

### Q3: On-policy distillation 为什么效果最好但最贵？

```
为什么效果最好:

  Response/Feature: student 学 teacher 的"固定输出/表示" (off-policy)
    → student 只能模仿, 探索空间被 teacher 限定
    → 无法超越 teacher, 也无法发现 teacher 没示范的好策略

  On-policy: student 在"自己的分布"上生成, teacher/reward 只给反馈
    → student 探索的是"自己能到达的状态空间"
    → 能发现 teacher 没教过的好路径 (R1 的 aha moment 就是这么来的)
    → 本质是 RL, 有探索性, 所以能超越 teacher

  类比:
    off-policy = 徒弟照师傅的菜谱做 (只能复刻)
    on-policy  = 徒弟自己炒, 师傅尝完打分 (能创新菜)

为什么最贵:

  1. 多轮迭代
     Response: 一次性造数据 → 训一次 (单轮)
     On-policy: 生成 → 打分 → 训练 → 再生成 → ... (多轮循环)

  2. 大量采样
     每个样本要 student 生成 N 条 (R1 里 group size G),
     只用好/坏的相对信号, 采样效率低

  3. 需要 reward
     规则 reward (R1): 要可验证的任务 (数学/代码), 通用任务没有
     神经 RM: 要训练 reward model, 又是额外成本 + hacking 风险

  4. 在线计算
     每轮都要 student 前向生成 (而非读静态数据集), GPU 开销大

成本对比 (直觉):
  Response:  $$  (API 费 + 一次训练)
  Feature:   $$$ (需要 teacher 权重, 但训练一次)
  On-policy: $$$$$ (多轮 × 大量采样 × 在线生成)

权衡:
  效果: On-policy > Feature > Response
  成本: On-policy > Feature > Response
  没有免费午餐 —— 效果和成本正相关。

  实战策略: 先用 Response/Feature 拿到 80% 的效果 (便宜),
           最后用 On-policy 精修最难的部分 (reasoning)。
           R1 就是这个思路: SFT 打底 → GRPO 攻 reasoning。
```

---

## 七、和我整体学习路径的衔接

```
CPT (week9-12)         SFT (week14-17)         蒸馏 (week18)         对齐 (week13)
─────────────          ─────────────           ────────────          ────────────
让模型理解             让模型会                让小模型继承            让模型合
领域语言               做任务                  大模型能力              人类偏好

                                    ┌─ 蒸馏在这里承上启下 ─┐
                                    │                      │
  CPT 给"语感"  →  蒸馏传"能力"(Self-Instruct/Zephyr dSFT)
                                          ↓
                                   蒸馏传"对齐"(Zephyr dDPO)  ←→  DPO/GRPO (week13)
                                          ↓
                                   蒸馏传"推理"(R1 on-policy)

week18 的位置:
  蒸馏是把前面 CPT/SFT 的成果"压缩传承"的技术,
  也是连接 SFT(week13之前) 和 对齐(week13) 的桥梁。

后续 (week19-21) 预期:
  - 蒸馏实战 (跑一个 response distill pipeline)
  - 评测 (对比 teacher/student)
  - 把蒸馏 + CPT + DPO 串成完整垂域管线
```

---

## 八、待深入的问题（留给实战）

- [ ] response distillation 在垂域的效果：用 GPT-4 造领域指令 vs 人工标注，ROI 对比？
- [ ] 蒸馏数据的"多样性 vs 质量"在垂域的权衡（Self-Instruct 平台期出现在 16K，垂域是否类似？）
- [ ] 能否把 feature distillation（需要权重）和 response distillation（只需 API）结合？混合蒸馏策略。
- [ ] Zephyr 的 AIF 在垂域的可行性：GPT-4 对领域回答的打分可靠吗？（可能不如通用领域）
- [ ] on-policy distillation 在垂域 reasoning 的应用（R1 范式如何迁移到医疗/法律推理）
