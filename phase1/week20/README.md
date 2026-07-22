# Week 20: 蒸馏深度大专题 — Feature(Logit KD)+ On-Policy 双主线

> **Phase1 最关键一段**。用户原话:「logit KD 也要做深,on-policy 也要,全部都要,缺什么补什么,不允许任何马虎、敷衍、偷懒」。
>
> 两个核心问题(week18 思考锚,本周干净回答):
> 1. **学 teacher 的 soft logit 分布(内部表征)vs hard argmax 输出(week19),差距在哪?** → Part A
> 2. **student 在自己分布上探索(on-policy)vs 学 teacher 分布(off-policy),差距在哪?能超越 teacher 吗?** → Part B

同 base(`week12_lora_cpt/50_50_fused`)、同数据(CMExam 2000 train)、同 eval 口径(CMExam holdout + CMMLU medical/general),与 week19 real/distill + week17 GRPO **逐字段直接对比**。

---

## 关键决策(探索 + 实测,不再改)

| 维度 | 选择 | 依据 |
|---|---|---|
| Teacher | **Qwen3-30B-A3B-Instruct-2507-MLX-4bit**,mlx_lm 直跑 | week19 验证稳定(~15 tok/s,holdout acc 0.865);bnb 4bit 不支持 Apple Silicon MPS(CUDA-only)→ MLX 是唯一稳定路径 |
| Student / base | Qwen3-1.7B,`week12_lora_cpt/50_50_fused` | 与 week15/17/19 同口径,delta 可比 |
| 框架 | torch + PEFT-LoRA(r16/α32/all-linear)+ MPS + bf16 | 与 week17/19 同栈;**fuse 走 PEFT `merge_and_unload`**(`run_dpo_eval.py:merge_lora`) |
| Eval | CMMLU:`week15/run_dpo_eval.py`;CMExam holdout:`week17/eval_cmexam.py` | base: CMExam 0.512 / 医 0.5663 / 通 0.6675 |

★ **vocab 一致性**:teacher(151936, qwen3_moe)== student(151936, qwen3)→ 离线 logit KD token 对齐可行(核心前提,实测 2000 条 0 mismatch)。

★ **两 venv**:student 训练/generate/eval 用 `phase1/.venv`(torch/transformers/trl/peft);teacher logits/judge 用 `~/Documents/GitHub/4bit-QLoRA-post-training/venv/`(mlx_lm)。

---

## Part A:Feature Distillation = Logit-Level KD

### 为什么 logit KD 而非 hidden state

传统 feature distill = 中间层 hidden state 对齐(DistilBERT + projection)。但**跨框架(MLX teacher + HF student)hidden state 不可行**:mlx.Tensor vs torch.Tensor、层命名/维度对齐复杂、跨框架 projection 不可导。

**现代生成式 LLM 的 feature distill 等价 = logit-level KD**(Hinton KD on token logits):学 teacher 每个 token 位置的**完整 soft 概率分布**,而非只 argmax(hard label = week19 distill 臂)。soft label 携带「类间关系」的 dark knowledge。结构性 insight:跨框架约束下,LLM feature distill 退化为 logit KD。

### KD loss(`kd_loss.py`,5 单测过)

```
L = α · CE(student, gold_token) + (1−α) · T² · KL_restricted(teacher_topk ‖ student_topk)
```
- CE:标准 next-token(prompt mask)
- KL restricted:teacher/student 分布都在 top-K(K=20)token 上 renormalize;**fp32 算**(bf16 数值不稳)
- 温度 T:两边 logits/T 后 softmax,梯度补偿 T²(Hinton)

### 三臂(对照 week19 distill hard-label)

| 臂 | α | T | 含义 |
|---|---|---|---|
| `kd_t2` | 0.5 | 2 | 经典 Hinton KD |
| `kd_t5` | 0.5 | 5 | 更软,暴露更多 dark knowledge |
| `kd_pure` | 0.0 | 2 | 纯 KL 不学 hard,看 soft 单独够不够 |
| 对照 week19 distill | 1.0 | — | 纯 CE on teacher argmax(hard label),已有结果 |

---

## Part B:On-Policy Distillation

### 设计

on-policy = student 在**自己分布**上生成 → 外部信号(teacher judge / rule reward)筛选 → 再学习。week18 理论:效果最好(student 探索自己分布,能发现 teacher 没示范的好路径,**能超 teacher** — R1 aha moment)但最贵。

Mac 约束:teacher MLX 不能在线产 logprob 给 TRL GKD/DistillationTrainer(后者 VLLMClient 绑 vLLM + 偏 on-policy generation)。故走 **rejection-sampling SFT(STaR / best-of-N)+ 可选 on-policy DPO**(industry standard,复用 MLX teacher judge + week17 reward + week15/16 SFT/DPO 栈,不依赖 vLLM)。

### 流程

1. `generate_student_samples.py` — student N=8/question 采样(temp=0.8,~16000 条,resume)
2. `judge_with_teacher.py` — teacher 30B 盲评 1-5(RLAIF,不给 gold 产生软信号,resume)
3. `prepare_onpolicy_data.py` — best-of-N 选择 → 3 SFT 数据 + dpo 偏好对(单测过)
4. train 3 SFT 臂(复用 week19 `train_distill_sft.py`)

### 三 SFT 臂(选择信号对比)

| 臂 | 选择信号 | 含义 |
|---|---|---|
| `rs_mcq` | rule correctness(letter==gold) | = GRPO 同信号,客观;全错回退 teacher 保 N=2000 |
| `rs_teacher` | teacher judge score | 盲评软信号,teacher 主观认同 |
| `rs_both` | correct ∩ teacher 精排 | rule 兜底 + teacher 精排 |
| `dpo_onpolicy`(stretch) | best vs worst 偏好对 | on-policy preference learning(week15 DPO 栈) |

---

## 文件

| 文件 | Part | 用途 |
|---|---|---|
| `extract_teacher_logits.py` | A | teacher MLX forward 存 top-K raw logits(resume) |
| `kd_loss.py` | A | α·CE + (1-α)·T²·KL restricted(fp32,5 单测) |
| `train_logit_kd.py` | A | Trainer + 自定义 compute_loss/collator + PEFT-LoRA |
| `run_feature.sh` | A | 三 KD 臂 detached 编排(train→fuse→eval→summarize) |
| `generate_student_samples.py` | B | student N=8 sampling(HF,temp=0.8,resume) |
| `judge_with_teacher.py` | B | teacher MLX judge 1-5 盲评(resume) |
| `prepare_onpolicy_data.py` | B | best-of-N → 3 SFT + dpo 数据(单测过) |
| `run_onpolicy.sh` | B | Part B detached 编排 |
| `summarize_week20.py` | 共享 | 汇总 → `week20_summary.json` + 大对照表(含 week19/GRPO) |
| `distill_feature.py` / `distill_on_policy.py` | — | 入口别名(指向真脚本,清掉旧 `[YOUR CODE]` stub) |

产物根:`phase1/results/week20_distill/{kd_t2,kd_t5,kd_pure,rs_mcq,rs_teacher,rs_both}/`(adapter)+ `{variant}_fused/`(HF)+ `data/` + `week20_summary.json`

---

## 跑(smoke-first 风闸)

```bash
# Part A(teacher logits 已提取 2000 条, top1==gold=0.892)
bash phase1/week20/run_feature.sh            # kd_t2/t5/pure → fuse → eval → summarize

# Part B
bash phase1/week20/run_onpolicy.sh           # generate → judge → prepare → train → eval → summarize

# 结果(任一 run 完成后刷新)
phase1/.venv/bin/python phase1/week20/summarize_week20.py \
    --sweep phase1/results/week20_distill \
    --runs kd_t2 kd_t5 kd_pure rs_mcq rs_teacher rs_both
```

---

## 结果

> **Part A + Part B 全完**(`run_feature.sh` + `run_onpolicy.sh` → `week20_summary.json`,六臂齐全)。
> 同 base(50_50_fused)/同 2000 CMExam 题/同 eval,与 week19 response + week17 GRPO 逐字段对比。
> Part B 全流程 16.2h:generate 5.6h(N=8×2000)→ judge 7.07h(16000 样本,1.59s/sample 实测)→ prepare → 3×SFT(63min/臂)→ fuse+eval+summarize。judge 分布 `[1:44.9%, 2:25.5%, 3:12.3%, 4:6.3%, 5:11.0%]` mean 2.13(parsed 16000/16000,非全-3,区分度好)。fallback:rs_mcq/rs_both 22.1%(correct 覆盖 0.779),rs_teacher 0%。

| 臂 | CMExam | Δ | CMMLU医 | Δ | CMMLU通 | Δ |
|---|---|---|---|---|---|---|
| base | 0.512 | — | 0.5663 | — | 0.6675 | — |
| kd_t2 (α.5 T2) | 0.524 | +0.012 | 0.5687 | +0.002 | 0.6850 | +0.018 |
| kd_t5 (α.5 T5) | 0.528 | +0.016 | 0.5625 | −0.004 | 0.6775 | +0.010 |
| **kd_pure (α0 T2)** | **0.544** | **+0.032** | 0.5675 | +0.001 | **0.6875** | **+0.020** |
| rs_mcq (rule) | 0.524 | +0.012 | 0.5763 | +0.010 | 0.6625 | −0.005 |
| rs_teacher (judge) | 0.510 | −0.002 | 0.5713 | +0.005 | 0.6800 | +0.013 |
| **rs_both (rule∩judge)** | 0.518 | +0.006 | **0.5787** | **+0.012** | 0.6700 | +0.003 |
| real(w19 hard) | 0.536 | +0.024 | 0.5413 | **−0.025** | 0.6625 | −0.005 |
| distill(w19 hard) | 0.530 | +0.018 | 0.5650 | −0.001 | 0.6750 | +0.008 |
| GRPO(w17) | 0.534 | +0.022 | 0.5687 | +0.002 | 0.6725 | +0.005 |

### Part A 结构性 insight(数据定稿)

1. **★ soft label 保知识,hard label 毁知识**(回答 week18 锚问「soft logit vs hard argmax 差距在哪」):week19 real(hard CE on gold)把 CMMLU 医学砸 **−0.025**(0.5663→0.5413);三臂 KD(soft KL)医学 Δ 全在 ±0.004 内,知识守住。**dark knowledge 验证**:teacher soft 分布携带类间关系,argmax 压成单 token → 在 teacher top1≠gold 的 ~11% 位置(train acc 0.892)CE 把 student 拉离正确次优,砸了广度知识。KD 三臂医学 Δ ≈ 0 是该现象最干净的对照。

2. **★ α=0 纯 KL 最佳,hard CE 是拖累**:kd_pure(完全不学 gold)CMExam **0.544(+0.032)全臂最高**,超 week19 real(+0.024)+ GRPO(+0.022);通识 0.6875(+0.020)也最高,医学守住(+0.001)。α=0.5 两臂 CMExam 只 +0.012/+0.016。解释:teacher(30B)比单 gold 答案更博学,teacher top1≠gold 时 CE 与 KL 打架;α=0 让 student 干净学 teacher 全分布,信 teacher 全分布 > 信单 gold。→ **强 teacher 蒸馏:丢掉 hard CE,纯 soft KL 够且更好。**

3. **温度次要,主轴是 α**:T2 vs T5 在 α=0.5 下互有胜负(CMExam T5 +0.016 略高,但医学 T5 −0.004 vs T2 +0.002、通识 T5 +0.010 vs T2 +0.018)。T∈[2,5] 影响小,经典 Hinton T=2 最稳;真正分水岭是 hard CE 开关(α),不是温度。

### Part B 结构性 insight(数据定稿)

1. **★ on-policy 没超 off-policy,但分工互补;单轮 STaR 超不了 teacher**(回答「on vs off 差距」+「能否超 teacher 0.865」):任务(CMExam)off-policy **kd_pure +0.032 全臂最高**,on-policy 三臂仅 +0.012/+0.006/−0.002 —— **单轮 best-of-N STaR 打不过学 teacher 全分布的 logit KD**。但 on-policy **rs_both 医学 0.5787(+0.012)全臂最高**(超 kd_pure 0.5675 / GRPO 0.5687 / real 0.5413):学 student 自己答对的医学推理,比学 teacher soft 分布更直接强化 domain 知识。**超 teacher 0.865?不能** —— 全部 rs_* 0.51-0.52,best-of-N 上限 = coverage 0.79(temp 0.8),oracle 都到不了 teacher;R1 式「超 teacher」要迭代自举 + 大 N + 真 RL,单轮 STaR 不具备。

2. **★ teacher judge 是 proxy reward,任务上反噬**(回答「judge vs rule 谁稳」):rs_teacher(judge 评分)CMExam **−0.002 唯一负臂**,而 rs_mcq(rule)+0.012。judge 评「解释质量」非「正确性」→ 自信的错答也能拿高分 → 选进来任务不涨反跌。**RLAIF 经典 proxy reward 失败模式**(reward hacking:优化代理非真值)。但 rs_teacher 通识 +0.013(最高)、fb=0(从不回退,覆盖全)—— judge 信号「覆盖广 + 通识好」但「任务不准」。rs_both(rule∩judge)取交集:rule 保正确、judge 保质量 → 医学全臂最高。**三信号互补,无单一最优。**

3. **★ correct 池里 letter 恒 = gold → rule 与 rule∩judge 只差解释质量**(最深隔离实验):rs_mcq 与 rs_both 的 chosen-letter 分布**完全相同** {A:278,B:333,C:331,D:349,E:268}。原因:correct 定义即 letter==gold,同一题的 correct 样本字母全 = gold → 无论按 rule(短)还是 judge(高分)选,字母恒定。**两 arm CMExam(0.524 vs 0.518)接近是结构性必然**,差异只在解释文本 → rs_both judge 重排的解释医学知识更好(医 0.5787 > rs_mcq 0.5763)。这隔离出「选择信号对解释质量的影响」:judge 选的解释 > 随机 correct,但 letter 已锁死、不影响任务分。

---

## 不在范围(边界)

- hidden-state distill(跨框架不可行,Part A 已论证退化原因)
- TRL 原生 DistillationTrainer / GKD(VLLMClient 绑 vLLM + 偏 on-policy,MLX teacher 对接风险高 + 与 week19 off-policy 不可干净对照)
- N>2000 / 全量(机制验通优先于规模)
- teacher 换外部 API(隐私冲突,本地 30B MLX 已够)
