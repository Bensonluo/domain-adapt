# Week14 实验矩阵 + 基线决策

> DPO/GRPO 实战的实验设计。本周**只设计不跑**（week15+ 执行）。矩阵按真实约束（0.8B / Apple Silicon / 已有资产 / 偏好数据 QC 发现）校准，不照搬 plan 文本里 3B 的假设。

---

## 真实约束（实验设计依据）

| 维度 | 现实 | 对实验的影响 |
|------|------|--------------|
| 基座 | `Qwen3.5-0.8B-Base-ms`（**不是** plan 里的 3B） | batch 小、reward 信号弱；β 敏感度高于大模型 |
| 偏好数据 | 1399 对（medical_evidence_DPO），**chosen 更长 93.5%**，32% 对长度差 >50% | **长度黑客风险高**：模型可能学"更长=更好"而非"更好=更好"。必须配长度控制的评估，考虑 length-normalized DPO 或 IPO |
| 序列长度 | prompt+chosen token p95=2980 / p99=3356 / max=3815 | DPO `max_length≈4096`（仅 54% 对 fit 2048）→ 0.8B 全量 + 4096 ctx 在统一内存上偏紧，需 grad-checkpoint |
| 参考模型 | DPO/GRPO 都要 frozen reference policy | 显存 ≈ 2× 模型参数（policy + ref）+ 激活；0.8B 可行但 GRPO 再加 G 次 generation 会更重 |
| 已有基线 | `week12_eval/real_cpt_fused`（CPT-only，已 fuse-ready） | CPT-only 基线立即可用；**CPT+SFT 基线不存在**（见下决策） |
| 评估管线 | `week12/_eval_core.py`（CMMLU medical_cn/general_cn）+ 偏好胜率 | 评估零新增基础设施 |

---

## SFT 缺口决策（本周最大产出之一）

week14 README 矩阵 + week15 `train_dpo.py` 都假设 "CPT+SFT" 基线，但 **Phase 1 从未跑过 SFT**（仅 phase0 有 SFT skill）。两条出路：

### 方案 A（推荐，默认）：首轮用 CPT-only 基线
- 基线 = `week12_eval/real_cpt_fused`（已 fuse-ready，本周即可起跑）
- DPO 直接打在 CPT 模型上。非标准（DPO 惯例打在 instruct/SFT 模型上），但：
  - **学习价值**：能看到"纯 CPT 模型 + 偏好对齐"的效应，与 week11/12 的 CPT-only 评估形成连续对照
  - **不阻塞**：无需先补 SFT 子管线
  - **风险**：base 模型未对齐过指令格式，DPO 可能更易发散（β 要偏小、lr 要保守）→ 这本身就是有价值的观察
- **若 DPO 不稳**（loss 爆 / 生成崩）→ 升级到方案 B

### 方案 B（备选）：先补 SFT，再 DPO
- week15 day0：用 `prep/build_sft_data.py`（需真实实现，非 stub）从已有医疗语料造 SFT 数据 + 用 phase0 的 SFT recipe（`phase0/week5/sft_trainer.py`、`phase0/week6/`）跑 SFT → fuse 出 `week15_sft_fused`
- 再在该 CPT+SFT 模型上跑 DPO。更标准、更稳，但多一个子管线（数据构建 + 训练 + fuse）

**本周默认走 A，决策权留给进入 week15 时定。** 矩阵两种基线都列（下表"基线"列标 A/B）。

---

## 实验矩阵

| # | 基线 | 方法 | 关键超参 | 预期信号 | 评估 |
|---|------|------|----------|----------|------|
| **0** | A:CPT-only / B:CPT+SFT | — （无对齐对照组） | — | 基线分；DPO/GRPO 的 delta 都相对它 | CMMLU med/general + 基线胜率 |
| **1** | 同上 | **DPO** | β=0.1（弱，贴近 ref） | 轻微偏好提升，遗忘最小 | 同上 + 偏好胜率 + 长度控制胜率 |
| **2** | 同上 | **DPO** | β=0.3（中） | 对齐/遗忘的甜点区候选 | 同上 |
| **3** | 同上 | **DPO** | β=0.5（强，偏离 ref） | 偏好提升最大但医疗可能崩 | 同上 |
| **4** | 同上 | **GRPO** | rule reward = 医学关键词 + 长度惩罚 + 结构分 | 验证 GRPO 在 0.8B 可跑；对比 DPO | 同上 |
| **5** | 同上 | **GRPO** | reward ablation（去掉长度惩罚 或 换纯格式 reward） | 量化 reward 各项贡献；**直接验证长度黑客**（去掉长度惩罚后是否模型输出变长） | 同上 + 平均输出长度对比 |

**β 扫描（行 1-3）** 回答"DPO 对齐强度 vs 通用遗忘的 trade-off 最优点"——week12 思考锚点的对齐版。

**额外建议（非强制第 7 行，视进度）**：在行 2（β=0.3）上换 **IPO loss** 或 **length-normalized DPO**，专攻本周 QC 发现的 93.5% 长度偏差——这是数据驱动的一手实验，比泛泛 β 扫更有 insight。

---

## 评估方式（复用 + 一项新增）

每行实验跑齐 3 类指标，全部相对基线（行 0）读 delta：

1. **领域保留**：CMMLU `medical_cn`（8 医学子集）平均 acc —— DPO 后医疗**不应崩**（week12 的 CPT 退化教训）。复用 `week12/_eval_core.py::run_mlx_evaluate`。
2. **通用遗忘**：CMMLU `general_cn`（4 非医学子集）平均 acc —— 对齐**不应牺牲**通用能力。
3. **偏好胜率**（对齐信号）：留出 ~100 对（从 train.jsonl 切 hold-out），DPO 模型对 chosen vs rejected 的 logprob，`P(chosen)>P(rejected)` 的占比。DPO 后应显著升。**纯 logprob，无需 LLM judge**。
4. **长度控制胜率**（长度黑客检测，本周 QC 驱动）：在胜率 3 里**按长度分桶**——若模型胜率提升完全来自"选更长的"，则是长度黑客。行 5（去长度惩罚）直接验证。

> 评估对象要求是 fuse 后的独立模型（`run_mlx_evaluate` 不吃 adapter，见 `_eval_core.py` docstring）。week15 DPO 产物 → `mlx_lm.fuse` → 再评估。

---

## 框架决策

- **主路线：TRL**（`DPOTrainer` / `GRPOTrainer`）——课程既定方向（week15 `train_dpo.py` + deep-dive-plan 代码块都用 TRL），week14 源码阅读（见 `trl_source_notes.md`）也对应。week15 需在 venv 装 `trl + torch + transformers`（PyTorch MPS 后端，0.8B 可行但偏慢）。
- **备选（Apple Silicon 友好）：MLX-DPO**。`mlx_lm` 无 DPO 子命令（已确认），但社区有 `mlx-tune`（SFT/DPO/ORPO on MLX）+ MLX-GRPO 实现。若 week15 TRL+MPS 太慢/不稳，切 MLX 路线。**本周不定，留 footnote**。

---

## 风险与缓解

| 风险 | 来源 | 缓解 |
|------|------|------|
| 长度黑客 | 数据 chosen 更长 93.5% | 行 5 + 长度控制胜率 + 可选 IPO/length-normalized 行 |
| DPO 发散（方案 A） | base 模型未对齐指令格式 | β 从 0.1 起、lr 保守、监控 loss；发散则升级方案 B（先 SFT） |
| 显存 | ref model = 2× params + 4096 ctx + GRPO G×generation | grad-checkpoint；GRPO 减 `num_generations`（G=4~8）；0.8B 在统一内存上可行 |
| GRPO generation 成本 | on-policy 每 prompt 生成 G 个 | 控 `num_generations` + 短 `max_new_tokens`；预期比 DPO 慢 3-5× |

---

## 复用资产清单（week15 执行时直接拿）

- 基线模型：`phase1/results/week12_eval/real_cpt_fused/`（方案 A）
- 偏好数据：`phase1/data/processed/preference/train.jsonl`（1399 对，本周产物；切 ~100 hold-out 给胜率评估）
- 评估管线：`phase1/week12/_eval_core.py`（`run_mlx_evaluate` / `fuse_model` / `TASK_GROUPS`）
- 评估数据：`phase1/data/cmmlu_local/`
- base tokenizer：`models/Qwen3.5-0.8B-Base-ms`（QC 已验证可用，vocab 248k）
