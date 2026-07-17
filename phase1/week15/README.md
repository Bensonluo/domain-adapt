# Week 15：DPO 实验（对齐强度 vs 通用遗忘的 β trade-off）

> 目标：在 week12 CPT baseline（`50_50_fused`，domain gain +0.043、通用无遗忘）上跑 **DPO**，
> 扫 β ∈ {0.1, 0.3, 0.5}，回答「对齐强度 vs 通用遗忘的 trade-off 最优点」，产出 DPO 模型 + 偏好胜率 + reward margin。
>
> **思考锚点**："DPO 的 beta 参数控制什么？beta 太大和太小各有什么后果？"

---

## 关键决策（全部查证，见 plan + week14/trl_source_notes.md）

| 决策 | 选择 | 依据 |
|------|------|------|
| 基线 | **CPT-only（方案 A）** = `week12_lora_cpt/50_50_fused` | Phase1 从未跑过 SFT；该基线已正 gain +0.043、fuse-ready、立即可跑（week14 矩阵决策） |
| 框架 | **TRL DPOTrainer + PyTorch/MPS + PEFT-LoRA** | 课程栈 = torch/trl/peft；week17 GRPO 复用同栈；`mlx_lm` 无 DPO 子命令 |
| dtype | **bf16**（不是 fp16） | torch 2.13 MPS 已支持 bf16（实测 matmul + autocast 均通）；bf16 指数域同 fp32，无 underflow、无需 GradScaler |
| reference | **`ref_model=None` + PEFT adapter-disable** | TRL 检测 `is_peft_model and ref_model=None` 时用 adapter-disable 当 reference（源码笔记 §1.3），0 额外显存 |
| 数据 | **1399 对**（medical_evidence_DPO），切 100 holdout + 1299 train；sweep 用 300-subset 先定方向 | week14 产物；chosen 更长 **93.5%** → 长度黑客风险，配长度分桶胜率监控 |

---

## 数据

- 源：`phase1/data/processed/preference/train.jsonl`（1399 对，TRL 原生 `{prompt,chosen,rejected}`）
- 切分（`prep_data.py`，seed 123，幂等）：`train_split.jsonl`（1299）+ `holdout.jsonl`（100，胜率评估用，不进训练）
- β-sweep：每 β 取前 300 对（`--limit 300`）先定 β 方向；胜出者后续可全量重训

---

## 训练

脚本：[`train_dpo.py`](train_dpo.py)（TRL DPOTrainer + PEFT-LoRA，离线 env 内嵌，落盘 `loss_log.csv` + `run_config.json`）
编排：[`run_dpo_sweep.sh`](run_dpo_sweep.sh)（3-β 串行，单 β 失败不中断，完成 marker = `adapter_model.safetensors`，detached）

```bash
# 单 β
phase1/.venv/bin/python phase1/week15/train_dpo.py \
  --model phase1/results/week12_lora_cpt/50_50_fused \
  --data phase1/data/processed/preference/train_split.jsonl \
  --beta 0.3 --lr 5e-6 --epochs 1 --limit 300 --max-length 2048 \
  --output phase1/results/week15_dpo/beta_0.3

# 3-β sweep（detached，仿 week12）
nohup bash phase1/week15/run_dpo_sweep.sh > /dev/null 2>&1 & disown
```

超参（跨 β 固定，唯一变量 = β）：lr 5e-6（LoRA-DPO 比 full-param 5e-7 高一个量级）/ 1 epoch / batch 1 / max_length 2048 / LoRA r16·α32·dropout0.05（与 CPT 对齐）/ loss=sigmoid / max_grad_norm 1.0。

### Smoke 结果（30 对，β=0.3，已通过）
loss 1.186→0.202，reward margin −0.656→+1.721，accuracy 0.2→1.0（step 20 内），save 路径全通。早期 grad_norm spike（372）正常，被 max_grad_norm 1.0 截断后恢复。

---

## β-sweep 结果

> 300-pair 子集 × 1 epoch × 3 β。base 与 DPO 同走 HF-route（HFLM+MPS）→ delta 有效。
> margin = β×Δlogratio（**跨 β 不可直比**，看漂移要除 β）。WR = holdout 100 对上 P(chosen)。

| β | 末步 margin | margin/β (漂移) | acc | medical_cn Δ | general_cn Δ | sum-WR | mean-WR | matched mean-WR |
|---|---|---|---|---|---|---|---|---|
| base | — | — | — | — | — | 0.01 | 0.27 | 0.154 |
| 0.1 | 13.86 | **139** (最远) | 1.0 | **−0.005** | +0.005 | 0.02 | 0.31 | 0.231 |
| 0.3 | 20.19 | 67 | 1.0 | −0.006 | +0.005 | 0.02 | 0.29 | 0.154 |
| 0.5 | 23.10 | **46** (最近) | 1.0 | −0.003 | +0.005 | 0.02 | 0.29 | 0.154 |

base 绝对分：medical_cn 0.566 / general_cn 0.668。

### 长度分桶（holdout 100 对，按 |Δlen|/max）

| bucket | n | base mean-WR | DPO(β=0.1) mean-WR |
|---|---|---|---|
| matched (<20%) | 13 | 0.154 | 0.231 |
| mid (20–50%) | 69 | 0.348 | 0.391 |
| skewed (>50%) | 18 | **0.056** | **0.056** |

sum-WR 在 mid/skewed 两档**全部 ≈ 0**（chosen 更长 → Σlogp 必输），仅 matched 13 对非零。

### 解读（不美化）

**① 没有灾难性遗忘——主安全检查通过 ✅**
medical_cn Δ 全在 −0.003~−0.006（week12 LoRA-CPT 噪声 ±0.04，这是噪声内），general_cn Δ 全 +0.005。DPO 打在 CPT baseline 上**没有重蹈 week12 全量 FT 的覆辙**（那次 medical −0.20、遗忘 +42%）。这是本轮最确定的正面结论。

**② CMMLU 几乎不动——符合预期，不是失败**
DPO 优化的是「偏好」（chosen vs rejected 整段回答），不是事实知识。CMMLU 是 4 选 1 事实召回，DPO 本就不该动它。三个 β 在 CMMLU 上不可区分 = 正常。

**③ holdout 胜率几乎没涨——DPO 严重过拟合 300 对 ⚠️（本轮核心发现）**
训练 acc=1.0、margin 飙到 +13~23（背完 300 对），但 holdout sum-WR 仅 0.01→0.02、mean-WR 0.27→0.31。**训练集 memorize ≠ holdout 泛化**：300 对 + LoRA 的高漂移容量 → epoch 0.3 就收敛 → 学到的是这 300 对的表面，不是「医疗证据偏好」的可迁移信号。三 β 在 holdout 上**统计不可区分**（n=100，差 0.02 在噪声内）。

**④ 长度偏差主导，DPO 没解决 ⚠️（week14 QC 预测，本周实证）**
week14 发现 chosen 更长 93.5% → 本周实测：sum-WR 全 ≈ 0（chosen 长必输 Σlogp）；skewed 档（长度差>50%）mean-WR 0.056——模型 95% 偏好更短的 rejected。这正是长度黑客风险的实锤，sigmoid DPO（Σlogp 目标）治不了。

### 选优（弱信号，诚实标注）

按预设门槛（medical_cn Δ ≥ −0.02 全过）+ matched-bucket mean-WR 排序，脚本选 **β=0.1**（0.231 > 0.154）。但这是 13 对上的 3/13 vs 2/13，**纯噪声**。三 β 在 holdout 不可区分，「最优 β」无统计意义。

### → week16 行动项（本轮负结果直接驱动）

1. **加数据**：300 对瞬间过拟合 → 用全量 1299（或更多），300 子集只够定方向不够泛化
2. **换 IPO / length-normalized DPO**（`loss_type="ipo"`）：长度偏差是头号失败，sigmoid Σlogp 治不了，IPO 的长度归一 score 直接对症
3. **`--noise` 失败模式**：钩子已埋，系统跑验证「数据噪声 vs 过拟合」的可分离性
4. β 扫描本身**已答完**：在 300 对 + sigmoid 设定下，β 不是关键变量（三 β 不可分），关键变量是数据量 + loss 形式

---

## 评估

| 脚本 | 作用 |
|------|------|
| [`run_dpo_eval.py`](run_dpo_eval.py) | PEFT `merge_and_unload` → HF-route CMMLU（`run_hf_evaluate` = HFLM+MPS，复用 `_eval_core` 的 TASK_GROUPS/cmmlu 本地 patch/score 解析）；base + 3 DPO 同 backend → delta 有效 |
| [`eval_winrate.py`](eval_winrate.py) | holdout 100 对胜率（sum-logp + mean-logp 双口径）+ 长度分桶（matched/mid/skewed） |
| [`summarize_dpo.py`](summarize_dpo.py) | 汇总表 + 选最优 β（门槛：medical_cn Δ ≥ −0.02；排序：matched-bucket mean-logp WR） |

任务组（复用 week12）：medical_cn（8 医学子集，领域保留）+ general_cn（4 非医学子集，通用遗忘）。limit=100，0-shot，seed 123。

---

## 验收清单

- [x] train_dpo.py 实现（TRL + PEFT，ref_model=None，bf16，离线 env）
- [x] smoke 通过（loss 降、margin>0、acc>0.5、save 路径通）
- [x] 数据切分（1299 train + 100 holdout）
- [x] 下游 eval/winrate/summarize 脚本就绪（compile + import + merge 路径全验证）
- [x] 3 个 β 训练完成（reward margin chosen>rejected，acc=1.0；drift 0.1>0.3>0.5 符合理论）
- [x] merge + CMMLU 评估（medical_cn Δ ≈ −0.005 噪声内 = 没崩；general_cn Δ +0.005 = 无遗忘）
- [x] holdout 胜率 + 长度分桶（**胜率几乎没涨 = 过拟合**；长度偏差主导 sum-WR≈0）
- [x] 选最优 β（β=0.1，但**弱信号**：三 β holdout 不可区分，详见解读）

---

## 衔接

- **week16**：IPO / length-normalized DPO（`loss_type="ipo"`，直接攻 93.5% 长度偏差）+ `--noise` 失败模式系统实验（钩子已埋）
- **week17**：GRPO（on-policy，独立栈；`beta=0` 不加载 ref，KL 用 k3 estimator）
- 最优 β 的 `beta_*_fused` 作为 week16/17 的对齐起点
