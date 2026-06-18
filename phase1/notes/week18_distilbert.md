# DistilBERT 论文精读笔记

> 论文: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (Sanh et al., Hugging Face, 2019)
> 链接: https://arxiv.org/abs/1910.01108
> Teacher: BERT; Student: DistilBERT
> PDF: [../week18/papers/DistilBERT_1910.01108.pdf](../week18/papers/DistilBERT_1910.01108.pdf)

---

## 核心观点 (TL;DR)

```
把蒸馏用在「预训练阶段」而不是「微调阶段」——
用大 BERT 当 teacher，蒸出一个层数减半的小 BERT 当 student，
靠一个 triple loss（distillation + MLM + cosine）训出来。

结果：
  - 参数量 -40%（110M → 66M）
  - 推理快 60%
  - 保留 BERT 97% 的语言理解能力（GLUE）

一句话：知识蒸馏的标准范式。把"学输出分布"和"学中间层表示"叠加，
小模型不学 hard label，而是学 teacher 的 soft 分布 + 隐藏状态方向，
这比单纯模仿输出学到的东西深得多。Feature distillation 的奠基之作。
```

---

## 核心问题

大模型（BERT、GPT-2）参数动辄上亿，部署到 edge/mobile 又慢又贵。能不能用一个小模型达到接近的性能？——压缩方法（剪枝、量化、蒸馏）里，蒸馏最自然：因为大模型已经学会的好东西，可以直接"传"给小模型。

**关键洞察**：之前的工作都是 task-specific 蒸馏（针对某个具体任务，如情感分类）。本文把蒸馏挪到 **预训练阶段**，蒸出一个通用的小模型，之后什么任务都能微调。

---

## 知识蒸馏（Knowledge Distillation）回顾

```
Hinton et al. 2015 的经典框架:

  Teacher (大、已训好)  ──┐
     输入 x → soft label t  │  t_i = softmax(z_i / T)
                          │         (T = temperature, 软化分布)
  Student (小、待训练)  ──┘
     输入 x → soft label s    s_i = softmax(z_i / T)

  蒸馏 loss: L_ce = Σ_i t_i * log(s_i)
           = teacher 和 student soft 分布的交叉熵

为什么有效?
  - hard label (one-hot) 只有 1 bit 信息: "答案是 class 3"
  - soft label 携带 teacher 的「不确定结构」:
    "答案大概率是 3，但 7 也有点像，1 几乎不可能"
  - 这些"暗知识"(dark knowledge) 是 teacher 的泛化能力，hard label 丢掉了
  - student 学 soft 分布 = 同时学正确答案 + 类间关系
```

**Temperature T 的作用**：T 越大，分布越平滑，暗知识越明显；训练时 teacher/student 用同一个 T，推理时 T=1 恢复标准 softmax。

---

## DistilBERT 的三件事

### 1. Student 架构：层数减半

```
BERT-base:     12 层 Transformer  ×  768 hidden
DistilBERT:     6 层 Transformer  ×  768 hidden   ← 每两层取一层

砍的是「层数」，不砍 hidden size。原因:
  - 现代 linear-algebra 框架对 hidden 维度的计算高度优化
  - 固定参数预算下，砍层数比砍 hidden 收益更大
  - token-type embedding 和 pooler 也删了（下游分类用不到/可重建）
```

### 2. Student 初始化：从 teacher 隔层拷贝

```
随机初始化 → 6 层 student 收敛困难（参数空间找不到好起点）

Trick: 因为 hidden 维度相同，直接从 teacher 的 12 层里
       隔一层取一层，作为 student 的初始权重。

  teacher layer 1  → student layer 1
  teacher layer 3  → student layer 2
  teacher layer 5  → student layer 3
  ...
  teacher layer 11 → student layer 6

消融显示：这一步贡献巨大（见下表），比 triple loss 里单个 loss 都重要。
```

### 3. Triple Loss：三个目标叠加

```
总 loss = L_ce (蒸馏) + L_mlm (MLM) + L_cos (cosine)

L_ce:  蒸馏主 loss，student 学 teacher 的 soft label 分布
       (Hinton 经典 KD loss，带 temperature)
L_mlm: 标准 BERT masked language modeling loss
       (student 自己也要会做填空，不只是抄 teacher)
L_cos: cosine embedding loss
       对齐 student 和 teacher 最后一层的 hidden state 方向
       (不只是输出分布，连内部表示的方向都对齐 → Feature distillation)
```

**三者的分工**：
- `L_ce` 学 teacher 的"判断"（soft label）
- `L_mlm` 学"自己生成"的能力（不能只抄不练）
- `L_cos` 学 teacher 的"内部表示方向"（这是 feature-level，比输出深一层）

---

## 关键实验与数字

### 性能对比（GLUE / IMDb / SQuAD）

| 模型 | GLUE 均分 | IMDb | SQuAD (EM/F1) | 参数量 | 推理时间(s) |
|------|----------|------|---------------|-------|------------|
| ELMo | 68.7 | - | - | 180M | 895 |
| BERT-base | 79.5 | 93.46 | 81.2/88.5 | 110M | 668 |
| DistilBERT | 77.0 | 92.82 | 77.7/85.8 | 66M | 410 |
| DistilBERT (D) | - | - | 79.1/86.9 | 66M | - |

- GLUE 保留 **97%** 性能（77.0 / 79.5）
- IMDb 只差 **0.64 个点**
- 推理快 **60%**（410s vs 668s）
- iPhone 7 Plus 上比 BERT 快 **71%**，整个模型才 207MB

### Ablation（GLUE 均分相对变化）

| 配置 | ΔGLUE |
|------|-------|
| 去 L_cos 和 L_mlm（只剩蒸馏） | -2.96 |
| 去 L_mlm（蒸馏 + cosine） | -1.46 |
| 去 L_cos（蒸馏 + MLM） | -0.31 |
| **完整 triple loss + 随机初始化** | **-3.69** |

**我的解读（重点）**：
1. **初始化最关键**：随机初始化直接 -3.69，比去掉任何一个 loss 都惨。说明"从一个好的起点出发"是蒸馏能成功的地基。
2. **cosine loss 贡献最小**（去掉只 -0.31），但作者仍保留，因为边际收益为正。
3. **MLM loss 有用**（去掉 -1.46），证明 student 不能纯抄 teacher，要有自己的目标。
4. **三个 loss 叠加 > 任何一个单独**，证实 triple loss 设计的必要性。

### 训练成本

```
数据: 和原版 BERT 一样（Wikipedia + BookCorpus）
算力: 8 × 16GB V100，约 90 小时
对比: RoBERTa 要 1024 × 32GB V100 训 1 天
结论: 蒸馏比从头预训练便宜一个数量级
```

---

## 为什么有效？（Feature distillation 的本质）

```
普通蒸馏 (response): student 学 teacher 的「输出概率分布」
  → 只学到 teacher "最后那一哆嗦" 的判断
  → 浅，student 不知道 teacher 内部怎么想的

Feature distillation (DistilBERT 加的 L_cos):
  student 学 teacher 的「中间层 hidden state 方向」
  → student 的内部表示空间被拉得和 teacher 对齐
  → 深一层：不只学结论，学"得出结论的过程"

类比:
  response 蒸馏 = 徒弟照抄师傅的菜谱（只学成品）
  feature 蒸馏 = 徒弟模仿师傅切菜的手势、火候的判断（学过程）
                 师傅的"内功"传给徒弟

为什么用 cosine 而不是 L2?
  - hidden state 的"方向"比"大小"更重要（语义在方向上）
  - cosine 对 magnitude 不敏感，更稳健
  - 不需要额外投影层（student/teacher 维度本就相同）
```

---

## 待深入的问题 / 与后续的衔接

- **为什么 DistilBERT 不需要 projection layer？** 因为 student 和 teacher 的 hidden size 相同（768），可以直接对齐。如果维度不同（如蒸到更窄的模型），就需要加一个可学的线性投影把 student 维度映射到 teacher 维度——这是后续 feature distillation 工作（如 TinyBERT）的做法。见 [../week18_distillation_comparison.md](week18_distillation_comparison.md) 自测题 2。
- **蒸馏 + 微调两段式**：DistilBERT 证明了"预训练阶段蒸馏"，下游任务还能再做一次"任务级蒸馏"（论文里 SQuAD 那行 DistilBERT (D)）。
- **和 Self-Instruct / Zephyr 的关系**：DistilBERT 是"有 teacher 权重"的 feature/response 蒸馏（能访问 hidden state）；Self-Instruct/Zephyr 是"只有 teacher API"的 response 蒸馏（只能拿到输出文本）。这是两种蒸馏范式的分水岭。

---

## 我的 takeaway

1. **蒸馏的本质是"信息密度"**：hard label 1 bit，soft label 携带整个分布的结构。把 teacher 学到的"类间关系"传给 student，比直接学正确答案高效得多。
2. **初始化是被低估的杠杆**。从 teacher 隔层拷贝权重，比随机初始化多保住 3.69 个 GLUE 点——这在工程上几乎免费，但收益比任何一个 loss 都大。做蒸馏时，"从 teacher 的好起点出发"是第一原则。
3. **triple loss 是 feature distillation 的模板**：response（学输出）+ 自身任务（学能力）+ feature（学表示）。后续所有 feature 蒸馏工作都是在这个框架上加料。
4. **DistilBERT 是蒸馏的"奠基"**，但它的 teacher 是同类小模型（BERT→小 BERT）。真正的范式跃迁是后面用 GPT-4 当 teacher 蒸到 7B（Zephyr）——把"大模型的能力"低成本搬到"可部署的小模型"。三篇串起来读才完整（见 [../week18_distillation_comparison.md](week18_distillation_comparison.md)）。
