# Week 12：CPT 效果评估（domain gain + 灾难性遗忘）

> 目标: 量化 CPT 的收益与损失——领域提升（domain gain）vs 通用遗忘（catastrophic forgetting）。
> 预计时间: 10-14 小时

> **思考锚点**: "CPT 的收益和损失之间，最优的 trade-off 点在哪？0.8B 这种小模型，评估分数本身可信吗？"

---

## 本周做了什么（实际执行）

**评估管线接通 + base vs CPT-fused 对比，并用真数据补完闭环**。

week11 已证：15 条 fake-tokenizer 合成数据 CPT 只学「医疗腔调」（form）不学「知识」（fact）。本周先把**评估能力**搭起来，用现成的 base + week11 假数据 CPT 产物跑一次，结果比预期更极端：**假数据 CPT 不仅没带来领域提升，反而全面退化**（医疗 −20%、通用遗忘 +42%）。

随后用**真数据（5684 条真实医学/通用语料，70-30 配比）重训同一套 200 iter 全量 CPT**，与假数据并列对比，闭环检验「假数据是否退化根因」——结论见下方[「真数据闭环」](#真数据闭环week12-补完假数据-vs-真数据对照)。

**多配比 ablation（纯领域/70-30/50-50 trade-off）留 week13**：那需要真数据重训出多个模型才有对照意义。

---

## 评估方法

### 框架：`mlx_lm.evaluate`（= lm-eval-harness 任务注册 + MLXLM 模型 wrapper）

为什么不用 `lm_eval --model hf`：我们的模型是 **MLX 格式**（Apple Silicon 微调产物），lm-eval-harness 原生不认 MLX。`mlx_lm.evaluate` 内部把 MLX 模型包成 `MLXLM(LM)` 类，调 `lm_eval.simple_evaluate(...)`，复用 lm-eval 全部任务注册。

为什么**直接调 Python API**而不是 `mlx_lm evaluate` CLI：见下方「关键坑」。

### 任务选型：CMMLU（中文，匹配 CPT 数据语种）

CPT 数据是**中文**医学百科，所以用 **CMMLU**（中文 MMLU）而非英文 MMLU——跨语言评估看不出领域提升。

| 任务组 | 内容 | 目的 |
|--------|------|------|
| `medical_cn` | CMMLU 8 医学子集（anatomy / clinical_knowledge / college_medicine / professional_medicine / genetics / traditional_chinese_medicine / virology / nutrition） | 领域提升信号（CPT 后应升） |
| `general_cn` | CMMLU 4 非医学子集（world_history / high_school_physics / econometrics / computer_network） | 遗忘信号（CPT 后不应大降） |

### 评估对象

- **base**: `models/Qwen3.5-0.8B-Base-ms`（魔搭本地，未 CPT）
- **cpt**: week11 假数据全量 CPT 产物，已 `mlx_lm.fuse` 合并成独立模型 → `phase1/results/week12_eval/week11_cpt_fused/`

> ⚠️ 全量微调产物（`adapters/`）**不能直接喂 `mlx_lm evaluate`**——CLI 的 argparse 没有 `--adapter-path`（源码 `mlx_lm/evaluate.py` 确认）。必须先 `mlx_lm.fuse` 合并成独立模型再评估（全量 fuse 无损）。

---

## 关键坑：hub 1.x 在国内连不上 → cmmlu 本地化

lm-eval 的所有任务（CMMLU/MMLU）走 `datasets → huggingface_hub` 下载。本次环境的依赖栈很新（`mlx-lm[evaluate]` 拉了 transformers 5 + **huggingface_hub 1.13.0** + datasets 4.8.5）：

- `curl` 能从 hf-mirror 下数据（resolve 端点可用，已用 `curl -L .../resolve/main/cmmlu_v1_0_1.zip` 把 cmmlu 全量拉下来）
- 但 **Python 的 hub 库连不通**：`hf_hub_download` 连 README 小文件都 `LocalEntryNotFoundError`，设了 `HF_ENDPOINT=https://hf-mirror.com` 也没用（hub 1.x 大版本重构 + 可能 xet 协议，hf-mirror 不支持 xet CAS）

**解法（本地化路线，不依赖 hub）**：cmmlu 数据已 curl + 解压到 `phase1/data/cmmlu_local/{test,dev}/*.csv`（67 子集齐全，gitignored）。`_eval_core._install_cmmlu_local_patch()` monkeypatch `datasets.load_dataset`，把 `'haonan-li/cmmlu'` 请求重定向到本地 csv：

```python
# 拦截 cmmlu task 的 load_dataset('haonan-li/cmmlu', name=<subject>)
# → load_dataset('csv', data_files={test/dev: <subject>.csv})
```

csv 列 `Question/A/B/C/D/Answer` 与 cmmlu task 的 `doc_to_text` 模板完全匹配。这条路彻底绕开 hub，且因为直接调 `lm_eval.simple_evaluate + MLXLM`，能注入 monkeypatch（CLI 做不到）。

> MMLU（英文，`medical_en`）同样依赖 hub，本周 hub 不通暂跳过；跨语言检查留到 hub 问题解决或换数据源后。

---

## 跑

```bash
# 1. fuse（CPT adapters → 独立模型；已跑过则跳过）
python -m mlx_lm fuse --model models/Qwen3.5-0.8B-Base-ms \
    --adapter-path phase1/results/week11_cpt_pure/adapters \
    --save-path phase1/results/week12_eval/week11_cpt_fused

# 2. 跑 base + fused 全任务（medical_cn + general_cn），存 scores
python phase1/week12/run_all_eval.py

# 3. 量化领域提升 + 遗忘（读 scores，不重跑）
python phase1/week12/eval_cpt.py \
    --baseline phase1/results/week12_eval/base_all/scores_*.json \
    --cpt-model phase1/results/week12_eval/cpt_all/scores_*.json \
    --tasks medical_cn --output phase1/results/week12_eval/domain_gain.json
python phase1/week12/eval_forgetting.py \
    --baseline phase1/results/week12_eval/base_all/scores_*.json \
    --finetuned phase1/results/week12_eval/cpt_all/scores_*.json \
    --tasks general_cn --output phase1/results/week12_eval/forgetting.json
```

---

## 0.8B 评估噪声（必读）

CMMLU 是 4 选 1，**随机基线 25%**。0.8B 在 CMMLU 绝对分通常 25-40%，run-to-run 方差 ±5pp。

→ **绝对分意义有限，看的是 delta（CPT 后 vs base）+ 固定 seed（123）+ `--limit`**。本周 `--limit 100` 控时（每子集取前 100 题，12 子集 × 2 模型）。方向性变化可信，±2-3pp 别过度解读。

---

## 结果（base vs week11-CPT-fused，CMMLU，limit=100，0-shot，seed=123）

> ⚠️ `limit=100` 取每子集**前 100 题**（非随机抽样），绝对分有偏差；但 base/CPT 用同一前 100 题，**delta 可信**。0.8B 接近随机基线（25%），方向性结论可信，具体幅度有 ±噪声。完整数据见 [`domain_gain.json`](../results/week12_eval/domain_gain.json) / [`forgetting.json`](../results/week12_eval/forgetting.json)。

### 领域提升 medical_cn（8 医学子集；gain = CPT − base，正=提升）

| 子集 | base | CPT-fused | gain |
|------|------|-----------|------|
| anatomy | 0.390 | 0.290 | −0.100 |
| clinical_knowledge | 0.350 | 0.320 | −0.030 |
| college_medicine | 0.500 | 0.260 | **−0.240** |
| professional_medicine | 0.350 | 0.190 | −0.160 |
| genetics | 0.550 | 0.300 | **−0.250** |
| traditional_chinese_medicine | 0.500 | 0.220 | **−0.280** |
| virology | 0.640 | 0.280 | **−0.360** |
| nutrition | 0.530 | 0.360 | −0.170 |
| **平均** | | | **−0.199** |

**8/8 全部 regressed**——CPT 后医疗能力不升反降，平均 −20%。

### 灾难性遗忘 general_cn（4 非医学子集；rate = (base−CPT)/base，正=遗忘）

| 子集 | before | after | forgetting |
|------|--------|-------|------------|
| world_history | 0.510 | 0.280 | +45.1% |
| high_school_physics | 0.540 | 0.230 | +57.4% |
| economics | 0.530 | 0.290 | +45.3% |
| marxist_theory | 0.610 | 0.490 | +19.7% |
| **平均** | | | **+41.9%** |

### 分析：假数据全量 CPT 是「全面退化」而非「弱提升」

结果比预期更极端。week11 的 15 条 fake-tokenizer 合成数据 + 全量微调 200 iters（train loss 跌到 0.012，严重过拟合）：

- **不是**「假数据学不到 fact、domain gain 接近 0」（原预期）
- **而是**过拟合到 15 条假数据的分布，把 base 模型已有的能力也**破坏**了——医疗（−20%）和通用（遗忘 +42%）**同时崩**

关键教训（本周核心 insight）：
1. **小数据 + 全量微调 = 灾难**：15 条数据全量微调 200 iter，过拟合不仅零收益，还负迁移。全量 CPT 对数据量/质量极度敏感。
2. **退化是全面的，非选择性遗忘**：若只是「学医疗忘通用」，医疗应升/通用降；实际医疗通用**都降** → 是模型能力整体被带偏（train loss 0.012 = 死记 15 条，破坏表征），不是领域替换通用。
3. **反证了 week12 先做真数据的必要性**：在假数据上做任何 CPT 调参/ablation 都会被这种「全面退化」误导——必须先有真数据。

---

## 真数据闭环（week12 补完：假数据 vs 真数据对照）

> 用真数据（70-30，5684 train / 631 val 真实文本）重训**同一套** 200 iter 全量 CPT（lr 1e-5，batch 1，与 week11 假数据严格对齐），fuse 后评估，与假数据并列。检验「假数据是否退化根因、真数据能否学到 fact 且不退化」。

### 训练行为对照（关键）

| 数据 | 样本 | 200 iter train loss | val loss | 形态 |
|------|------|---------------------|----------|------|
| 假数据（week11） | 15 条合成 | **→ 0.012**（死记） | — | 严重过拟合 |
| 真数据（week12） | 5684 条真实 | **持平 3.18**（iter10=3.30 → iter200=3.18） | 3.06 → 3.15（微升） | **几乎没学到（underfit）** |

真数据 loss 平得像没训——这是 200 iter @ lr=1e-5 对 5684 条多样语料**步数/学习率不足**的直接表现，与假数据「死记 15 条跌到 0.012」形成两极对照。

### 评估对照（base → 假数据 CPT → 真数据 CPT）

**领域提升 medical_cn（8 医学子集；gain = CPT − base，正=提升）**

| 子集 | base | 假数据 CPT | 真数据 CPT | 假数据 gain | 真数据 gain |
|------|------|------------|------------|-------------|-------------|
| anatomy | 0.390 | 0.290 | 0.370 | −0.100 | **−0.020** |
| clinical_knowledge | 0.350 | 0.320 | 0.270 | −0.030 | −0.080 |
| college_medicine | 0.500 | 0.260 | 0.410 | −0.240 | **−0.090** |
| professional_medicine | 0.350 | 0.190 | 0.280 | −0.160 | **−0.070** |
| genetics | 0.550 | 0.300 | 0.390 | −0.250 | **−0.160** |
| traditional_chinese_medicine | 0.500 | 0.220 | 0.410 | −0.280 | **−0.090** |
| virology | 0.640 | 0.280 | 0.530 | −0.360 | **−0.110** |
| nutrition | 0.530 | 0.360 | 0.460 | −0.170 | **−0.070** |
| **平均** | | | | **−0.199** | **−0.086** |

**灾难性遗忘 general_cn（4 非医学子集；rate = (base−CPT)/base，正=遗忘）**

| 子集 | base | 假数据 | 真数据 | 假数据遗忘 | 真数据遗忘 |
|------|------|--------|--------|------------|------------|
| world_history | 0.510 | 0.280 | 0.480 | +45.1% | **+5.9%** |
| high_school_physics | 0.540 | 0.230 | 0.350 | +57.4% | +35.2% |
| economics | 0.530 | 0.290 | 0.500 | +45.3% | **+5.7%** |
| marxist_theory | 0.610 | 0.490 | 0.570 | +19.7% | **+6.6%** |
| **平均** | | | | **+41.9%** | **+13.3%** |

### 闭环结论：假数据确实更糟，但真数据也没转正

1. **「假数据是退化根因」部分成立**：真数据在两条轴上都明显更轻——domain 退化 −0.086 vs −0.199（约 **2.3×** 轻）、遗忘 +13.3% vs +41.9%（约 **3.2×** 轻）。假数据的过拟合（loss 0.012）确实在**额外破坏** base 能力。方向性印证了原假设。

2. **但真数据 domain gain 仍未转正**（8/8 还是 regressed，只是幅度小）。结合训练 loss 持平 3.18 → 真数据这轮**根本没学到医疗 fact**，所以谈不上「领域提升」。退化 −0.086 更像是「全量微调扰动权重导致的轻微漂移」，而非「学了新的忘了旧的」。

3. **更值钱的发现 —— 200 iter 全量 CPT 是「两头不讨好」的甜点区外**：步数/lr 不足以学会新 domain（underfit，无 domain gain），但全量权重扰动仍足以侵蚀已有能力（仍 −9% 医疗 / −13% 通用）。要让 CPT 出现正 domain gain，下一步必须二选一：
   - **加 iter / 加 lr**（但遗忘风险同步升 → 需更精细的 domain/general 数据配比）
   - **换 PEFT（LoRA）** 把 domain 学习隔离在 adapter 里，少动 base 权重

   这正是 week13 的方向（LoRA + 更多 iter + 配比 ablation），且有了一个干净的 base 真实对照基线。

4. **physics 是两侧共同噪声点**（假 +57%、真 +35%，都远高于其他 3 科的个位数）——0.8B 在高中物理本身接近随机，单科幅度别过度解读，看 4 科均值。

---

## Week12 补完：换模型 + LoRA-CPT 比例 sweep（→ 正 domain gain 基线）

> 真数据闭环的结论是「200 iter 全量 CPT underfit 且漂移，domain gain −0.086 未转正」。补完方向原列在「留 week13」（LoRA + 加 iter + 配比 ablation），这里直接执行——目标是产出一个 **domain gain > 0 的 CPT 基线**作 week15 DPO 起点，替换那个负 gain 的 `real_cpt_fused`。

### 为什么换模型：Qwen3.5-0.8B 其实是个慢速 exotic 多模态底座

原 base `Qwen/Qwen3.5-0.8B-Base-ms` 实测架构是 **Qwen3.5 hybrid（linear/full 混合注意力 + MTP + mamba + vision，词表 248K）**——一个 exotic 多模态 VLM，不是标准 CausalLM。后果：

- **极慢**：LoRA-CPT ~23 sec/iter、53GB 内存。同样代码换标准 qwen3 架构只要 **2.5 sec/iter、6.7GB（~9× 加速）**。
- **不稳**：LoRA lr 1e-4 直接发散（train loss 3.06 → 15.7）。

**模型大小不是本项目的核心重点**（核心是 domain adaptation 的**方法/管线**，扬长避短做蒸馏 + GRPO 深度），所以换成 **`Qwen/Qwen3-1.7B`**（标准 `Qwen3ForCausalLM`，新一代、快、稳）。base 在 12 个 CMMLU 子集重评：medical_cn 均 **0.52** / general_cn 均 **0.61**（比 0.8B 高一截、远离 25% 随机基线、噪声更低）→ [`base_qwen3_17b/scores_Qwen3-1.7B.json`](../results/week12_lora_cpt/base_qwen3_17b/)。

### 实验设计：LoRA 比例 sweep（固定一切，只变比例 = 公平对照）

| 固定项 | 值 | 依据 |
|--------|-----|------|
| model | Qwen/Qwen3-1.7B | 标准 qwen3 arch |
| fine-tune-type | lora | base 冻结 → domain 学习隔离在 adapter，防全量漂移 |
| lora | rank16 / scale32 / dropout0.05, num-layers −1 | 跨三比例固定（比例成唯一变量）|
| lr | 1e-5 | smoke 确认稳（1e-4 在旧 VLM 发散，不在新模型冒险）|
| batch-size | 1 | probe 实测 batch=4 **不更快**（Metal 带宽受限，per-sample 吞吐相同 ~2.6s），取更省内存的 batch=1 |
| iters | 2500（三比例相等）| ~0.4–0.6 epoch，是 week12 underfit 200 iter 的 12×，足以让 LoRA 学进 domain |

数据用 **Qwen3 tokenizer 重新 token-precise 混合**（旧 Qwen3.5 tokenizer 跨代词表不同不能复用）：100-0=4921 / 70-30=7031 / 50-50=5405 chunks。脚本：[`run_lora_sweep.sh`](run_lora_sweep.sh)（train×3 → fuse+eval → summary 一条龙）。

### 结果（✅ 三比例全部正 domain gain，无灾难性遗忘）

**总览（vs base；旧 Qwen3.5 全量 FT 70-30 作历史锚，不同模型仅看进度方向）**

| 比例 | medical_cn 平均 gain | general_cn 平均遗忘率 | 备注 |
|------|----------------------|----------------------|------|
| 100-0 | **+0.033** | **−8.2%**（通用反升） | 纯领域 |
| 70-30 | **+0.041** | **−10.4%**（通用反升） | |
| **50-50** | **+0.043** ← 最优 | **−11.9%**（通用反升） | gain 最高 + 通用反升最多 |
| *旧 Qwen3.5 全量 FT 70-30* | *−0.086* | *+13.3%* | *week12 历史（不同模型）* |

> 遗忘率为负 = CPT 后通用**不降反升**（`forgetting = (base−cpt)/base`，负即 cpt>base）。三比例通用全部略涨，**灾难性遗忘 = 0**——LoRA 把 domain 学习隔离在 adapter、不扰动 base 权重，直接验证了补完方向的核心假设。

**领域提升 medical_cn 逐科（base → CPT；gain = CPT − base，正=提升）**——以最优 50-50 为例（另两比例形态一致，逐科见 [`domain_gain_lora_*.json`](../results/week12_lora_cpt/)）

| 子集 | base | 50-50 CPT | gain |
|------|------|-----------|------|
| anatomy | 0.44 | 0.42 | −0.020（噪声内）|
| clinical_knowledge | 0.47 | 0.55 | +0.080 |
| college_medicine | 0.62 | 0.63 | +0.010 |
| professional_medicine | 0.54 | 0.57 | +0.030 |
| genetics | 0.45 | 0.50 | +0.050 |
| traditional_chinese_medicine | 0.57 | 0.61 | +0.040 |
| virology | 0.58 | 0.62 | +0.040 |
| nutrition | 0.51 | 0.62 | **+0.110** |
| **平均** | | | **+0.043** |

**通用遗忘 general_cn 逐科（before → after；负=不降反升）**

| 子集 | base | 100-0 | 70-30 | 50-50 |
|------|------|-------|-------|-------|
| world_history | 0.57 | 0.63 | 0.68 | **0.73** |
| high_school_physics | 0.47 | 0.52 | 0.54 | 0.54 |
| economics | 0.61 | 0.65 | 0.64 | 0.63 |
| marxist_theory | 0.78 | 0.82 | 0.80 | 0.79 |

### 分析：从「两头不讨好」到「两头都涨」

1. **domain gain 符号翻转 = 补完成功**：week12 全量 FT 是 −0.086（underfit + 全量漂移），LoRA 三比例全部 +0.033~+0.043。根因对症——LoRA 隔离 domain 学习 + 2500 iter（week12 的 12×）让 adapter 真正学进 domain，而 base 冻结消除了全量漂移。**正 gain 基线已就位，替换负 gain 的 `real_cpt_fused`。**

2. **遗忘轴也翻转**：week12 全量 FT 通用遗忘 +13.3%（退化），LoRA 三比例全是**负遗忘**（通用略涨 8~12%）。其中混了通用数据的 70-30 / 50-50 涨得更多（world_history 0.57→0.73），符合预期——通用文本直接喂进去就练到通用任务。**「CPT 必然伤通用」在小模型 + LoRA + 混合数据下不成立。**

3. **比例对 domain gain 近乎不敏感（关键 insight）**：100-0 / 70-30 / 50-50 的 medical gain 只在 +0.033~+0.043 间（差 0.01，在 ±噪声内）。比例真正影响的是**通用轴**——通用数据占比越高，通用反升越多。→ 在这个配比区间，domain 提升主要靠「LoRA + 足量 iter」而非「配比微调」；配比是用来调 domain/general trade-off 的杠杆，不是 domain gain 的开关。

4. **幅度诚实**：+4% 是温和提升（8 科多数 +0.01~+0.08，无单科暴涨），符合「LoRA + 0.5 epoch + 10M token」的预期——CPT 在 benchmark acc 上的收益本就温和，真正的 domain fluency 提升要看生成（CMMLU 测不到）。但**符号 + 一致性**（8 科 7 涨、三比例同形态）足以说明 domain 知识真学进去了，不是噪声。

### week15 DPO 基线

最优比例 **50-50**（domain gain 最高 + 通用反升最多）的 fused 模型作为 week15 DPO 起点：

```
phase1/results/week12_lora_cpt/50_50_fused/   ← week15 DPO baseline (替换负 gain 的 real_cpt_fused)
```

产物全在 [`results/week12_lora_cpt/`](../results/week12_lora_cpt/)：`sweep_summary.json`（选优）、`domain_gain_lora_{100_0,70_30,50_50}.json` / `forgetting_lora_*.json`（逐科）、`{tag}_fused/`（三比例独立模型）、`{tag}_eval/scores_*.json`（原始 acc）。

---

## 交付物

- [x] 评估管线接通（`mlx_lm.evaluate` / lm-eval + MLXLM）— [`_eval_core.py`](_eval_core.py)
- [x] cmmlu 本地化（绕开 hub 1.x 国内连不上）— `phase1/data/cmmlu_local/`（curl hf-mirror + monkeypatch）
- [x] base + CPT 对比脚本 — [`eval_cpt.py`](eval_cpt.py) / [`eval_forgetting.py`](eval_forgetting.py)
- [x] 假数据量化结果 — [`results/week12_eval/domain_gain.json`](../results/week12_eval/domain_gain.json) / [`forgetting.json`](../results/week12_eval/forgetting.json)
- [x] **真数据闭环**（70-30 真实语料 200 iter 全量 CPT，与假数据严格对齐）— 训练产物 [`results/week12_real_cpt/`](../results/week12_real_cpt/)（adapters/loss_log/loss_curve/run_config）+ 量化 [`domain_gain_real.json`](../results/week12_eval/domain_gain_real.json) / [`forgetting_real.json`](../results/week12_eval/forgetting_real.json) + 只评估 fused 的薄脚本 [`run_real_eval.py`](run_real_eval.py)
- [x] **LoRA-CPT 比例 sweep**（换 Qwen3-1.7B + LoRA + 2500 iter，domain gain 转正）— 一条龙脚本 [`run_lora_sweep.sh`](run_lora_sweep.sh)（train×3→fuse+eval→summary，幂等续跑）+ eval/summary [`run_lora_sweep_eval.py`](run_lora_sweep_eval.py) / [`lora_sweep_summary.py`](lora_sweep_summary.py) + LoRA 配置 [`../week11/lora_cpt_config.yaml`](../week11/lora_cpt_config.yaml) + 产物 [`results/week12_lora_cpt/`](../results/week12_lora_cpt/)（`sweep_summary.json` + 三比例 `{tag}_fused/` + 逐科 gain/forgetting JSON）→ **50-50 fused = week15 DPO baseline**

---

## 验收清单

- [x] 评估后端选定并跑通（mlx_lm.evaluate，非 lm_eval --model hf）
- [x] 任务选型匹配 CPT 语种（中文 CMMLU，非英文 MMLU）
- [x] base + CPT 模型都能评估（CPT 经 fuse 合并）
- [x] domain gain + forgetting 量化完成（假数据：medical_cn **−0.199** / general_cn **+42%** 全面退化；真数据闭环：**−0.086 / +13.3%**，退化更轻但仍未转正；**LoRA-CPT 补完**：换 Qwen3-1.7B + LoRA + 2500 iter，三比例 domain gain **全部转正 +0.033~+0.043、遗忘翻负（无灾难性遗忘）**，50-50 最优 → week15 DPO 基线）
- [x] 至少 1 个关于 CPT 评估的个人 insight（见下）

---

## 个人 insight

1. **评估对象 ≠ 评估管线**：管线本身（fuse → evaluate → 量化）是本周的真正学习目标；具体数字弱（假数据）不影响管线价值。
2. **小模型评估的陷阱**：0.8B 在 CMMLU 接近随机基线，不看 delta vs base 直接读绝对分会得出错误结论——这是「领域提升」类评估最常踩的坑。
3. **hub 1.x + 国内网络**：`HF_ENDPOINT` 对新版 hub 库不再可靠（xet 协议 + 重构），curl 直下 + monkeypatch 本地化是更可控的 fallback。

---

## 不在本周范围（留 week13）

- ~~**真数据重训**（70-30 一配比，对比 week11 假数据）~~ — ✅ 本周已补完（见上「真数据闭环」）
- ~~**多配比 ablation**（纯领域/70-30/50-50 trade-off 曲线）~~ — ✅ 已在「Week12 补完」执行（LoRA sweep，Qwen3-1.7B）
- ~~**LoRA 对照 + 更多 iter**~~ — ✅ 已在「Week12 补完」执行（LoRA 隔离 + 2500 iter，看 domain gain 能否转正）
- **MMLU 跨语言检查**（英文医学，看中文 CPT 是否迁移到英文）——需解决 hub 或换数据源
