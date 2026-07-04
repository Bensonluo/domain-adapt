# TRL 源码阅读笔记 — DPOTrainer & GRPOTrainer（实现层）

> 数据来源: `huggingface/trl` `main` 分支, `trl/trainer/dpo_trainer.py` (~1700 行) 与 `trl/trainer/grpo_trainer.py` (~2900 行)。
> 本笔记只描述**代码如何实现**，不重推数学（week13 已完成：[`derivation_dpo_detailed.md`](../week13/derivation_dpo_detailed.md)）。
> 读码时点：`selective_log_softmax` 是 TRL 内部的 numerically-stable gather 版 `log_softmax`，全文件 logprob 都用它。

---

## 1. DPOTrainer 实现 (`trl/trainer/dpo_trainer.py`)

### 1.1 入口：`compute_loss` → `_compute_loss`

HF `Trainer` 在每个训练步调 `compute_loss(model, inputs, ...)`。DPOTrainer 的 `compute_loss` (L1682) 只做派发：

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if self.use_liger_kernel:
        return self._compute_loss_liger(model, inputs, return_outputs)
    return self._compute_loss(model, inputs, return_outputs)
```

> **重要事实**：当前 TRL main **没有** `dpo_loss` / `dp_loss` / `get_batch_logps` 这些方法（老版教程里有）。新版把所有逻辑塞进 `_compute_loss` (L1321) 一个函数里，per-token logprob 直接 inline 算，不再抽函数。

### 1.2 per-token logprob → 序列 logprob (L1336-1347)

```python
shift_logits = outputs.logits[..., :-1, :]
shift_labels = input_ids[..., 1:]
shift_completion_mask = completion_mask[..., 1:]
per_token_logps = selective_log_softmax(shift_logits, shift_labels)
per_token_logps[shift_completion_mask == 0] = 0.0   # 屏蔽 prompt + padding
if self.ld_alpha is None:
    logps = per_token_logps.sum(dim=1)               # 序列 logprob = Σ logp(token)
chosen_logps, rejected_logps = logps.chunk(2, dim=0) # batch 拼成 [chosen, rejected]
```

关键点：
- **batch 顺序写死** `[chosen, rejected]`，靠 `chunk(2)` 拆——这是 TRL DPO 的隐性 contract，数据 collator（`DPODataCollatorWithPadding`, L152）必须保证顺序。
- **屏蔽靠 `completion_mask`**：prompt 和 padding 位置的 logprob 直接清零再求和，等价于只在 completion token 上求和。
- `ld_alpha` 是 SIMPO-style 的「共享前缀 vs 尾巴」长度归一（默认 `None`，不影响标准 DPO）。

### 1.3 参考策略处理 (L1355-1389) — 三选一

`__init__` (L861-878) 根据 config 决定 reference 怎么来：

| 模式 | 触发条件 | 代码行为 |
|---|---|---|
| **预计算** | `args.precompute_ref_log_probs=True` | `_precompute_ref_logps` (L1122) 预先跑一遍 ref 模型，把 `ref_chosen_logps`/`ref_rejected_logps` 写回 dataset，训练时直接 `inputs["ref_chosen_logps"]` (L1361)，**不占显存**。 |
| **PEFT adapter** | `is_peft_model(model) and self.ref_model is None` | `with use_adapter(model, adapter_name="ref" if "ref" in peft_config else None)`（L1373-1376）：禁用 adapter → 基模 = reference。**0 额外显存**。 |
| **独立 ref 模型** | 其它（全参微调） | `self.ref_model = create_model_from_path(ref_model_path, ...)` (L876)——**完整第二份权重**。 |

非预计算路径的 ref forward 全部在 `torch.no_grad()` + `disable_gradient_checkpointing` 下跑（L1371）。

### 1.4 损失表达式 (L1392-1428) — 与 week13 公式逐项对应

```python
chosen_logratios   = chosen_logps   - ref_chosen_logps     # = log(π_θ(y_w)/π_ref(y_w))
rejected_logratios = rejected_logps - ref_rejected_logps   # = log(π_θ(y_l)/π_ref(y_l))

if self.f_divergence_type == "reverse_kl":   # 标准 DPO
    chosen_scores   = chosen_logratios
    rejected_scores = rejected_logratios
...
delta_score = chosen_scores - rejected_scores

# 标准 sigmoid (DPO) loss:
per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
```

**与 week13 公式的对应**（不重推，只标号）：

| DPO 公式项 | 代码 |
|---|---|
| `log π_θ(y_w) − log π_ref(y_w)` | `chosen_logps − ref_chosen_logps` → `chosen_logratios` |
| `log π_θ(y_l) − log π_ref(y_l)` | `rejected_logps − ref_rejected_logps` → `rejected_logratios` |
| `β · [Δw − Δl]` | `self.beta * (chosen_scores - rejected_scores)` = `self.beta * delta_score` |
| `−log σ(β · Δ)` | `-F.logsigmoid(self.beta * delta_score)` |

`self.beta` 就是 DPO 公式里那个 `β`（KL 约束强度），在 `__init__` L731 从 `args.beta` 读入。

### 1.5 多 loss 钩子（同一个 `_compute_loss` 里分支）

`self.loss_types` 是个 list（L733），支持混权：`sigmoid` (DPO)、`hinge`、`ipo`、`exo_pair`、`nca_pair`、`robust`、`bco_pair`、`sppo_hard`、`aot`、`aot_unpaired`、`apo_zero`、`apo_down`。

**IPO 分支** (L1440-1448) 顺手记一下，因为它跟 DPO 形式最像，且**直接对应本周 QC 发现的长度偏差问题**：

```python
chosen_avg_score   = chosen_scores   / chosen_mask.sum(dim=1).clamp(min=1.0)
rejected_avg_score = rejected_scores / rejected_mask.sum(dim=1).clamp(min=1.0)
ipo_delta = chosen_avg_score - rejected_avg_score
per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2  # (Eq.17), β 在 IPO 里是 τ
```

注意 IPO 用的是**长度归一化的 score**（除以 completion token 数），跟 sigmoid DPO 用 `Σ logp` 不同——这是数值稳定性必需，paper 里没明说（TRL 注释专门提了一句 "confirmed with IPO authors"）。

> **本周 QC 联动**：[`pref_qc_report.json`](pref_qc_report.json) 发现 chosen 更长占 93.5%、`|diff|/max>0.5` 占 32%——长度黑客风险高。sigmoid DPO 用 `Σ logp`（chosen 更长 → logp 总和更大 → 假性偏好），**IPO / length-normalized DPO 是天然的缓解**。这一点已写进 [实验矩阵](experiment_matrix.md) 行 2 的可选第 7 实验。

---

## 2. GRPOTrainer 实现 (`trl/trainer/grpo_trainer.py`)

### 2.1 Group 采样：`RepeatSampler` (L1085-1092)

```python
return RepeatSampler(
    data_source=dataset,
    mini_repeat_count=self.num_generations,                       # = G，每个 prompt 连采 G 次
    batch_size=self.args.generation_batch_size // self.num_generations,
    repeat_count=self.num_iterations * self.args.steps_per_generation,
    ...
)
```

- `num_generations` 就是论文里的 **G**（L634）。
- sampler 把同一个 prompt 在 batch 里**连续重复 G 次**，下游所有 `view(-1, num_generations)` 都依赖这个布局。

### 2.2 生成：`_generate_single_turn` (L1515) → `_generate_and_score_completions` (L2002)

`generation_config`（L943-949）默认是 **on-policy sampling**：

```python
{"do_sample": True, "temperature": self.temperature,
 "top_p": self.top_p, "top_k": self.top_k}
```

- 路径有三条：vLLM (L1521)、`generate_batch` 连续批处理 (L1551)、`unwrapped_model.generate(...)` 标准路径 (L1589)。
- 全部包在 `torch.no_grad()` + `unwrap_model_for_generation` 里——生成阶段不更新参数、不存梯度。

### 2.3 Reward 拿到的方式 (L460-479)

`reward_funcs` 接受三种形式，统一塞进 list：
1. **string** → 当成 HF repo id，`AutoModelForSequenceClassification.from_pretrained(..., num_labels=1)` (L472) — 学到的 reward model。
2. **`nn.Module`** → 直接当 reward model 用。
3. **callable** → 用户自定义规则函数（最常见，例如数学题的正确性判分）。

多个 reward 在 L2394 加权和：`rewards = (rewards_per_func * self.reward_weights).nansum(dim=1)`。

### 2.4 Reward 归一化 / group-relative advantage (L2394-2418) — **GRPO 的核心**

```python
rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
if self.scale_rewards == "group":
    std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)
    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
elif self.scale_rewards == "batch":
    std_rewards = nanstd(rewards).expand_as(rewards)
advantages = rewards - mean_grouped_rewards
if self.scale_rewards != "none":
    advantages = advantages / (std_rewards + 1e-4)
```

**这就是 `(R − mean_group) / std_group` 那一行公式**，逐项：
- `rewards.view(-1, num_generations)`：把 (B*G,) reshape 成 (B_prompts, G) — 每行一个 prompt 的 G 个 completion 的 reward。
- `nanmean(dim=1)`：每组 G 个 reward 的均值 = baseline。
- `repeat_interleave`：把 (B_prompts,) 的均值广播回 (B*G,)，与原 rewards 对齐。
- `1e-4` 防 std=0（所有 completion 同分）。

`scale_rewards` 默认 `"group"`（per-prompt 归一），可选 `"batch"`（全局归一，类似 RLOO）或 `"none"`（只减均值不除 std）。

### 2.5 无 critic 的证据

```
grep -E 'critic|value_head|value_network|V\(s\)|value_model|baseline.*network' grpo_trainer.py
# (空输出)
```

**整个 GRPO trainer 没有 value/critic 网络**。`advantages` 完全由 `rewards` 的 group 统计量（mean/std）算出，baseline 就是 `mean_grouped_rewards`——这正是 GRPO 对 PPO 的核心简化。对应 week13 笔记「GRPO 去掉 critic，用 group mean 当 baseline」。

### 2.6 PPO-style clipped surrogate loss (L2767-2780)

```python
old_per_token_logps = inputs.get("old_per_token_logps") or per_token_logps.detach()
log_ratio = per_token_logps - old_per_token_logps
coef_1 = torch.exp(log_importance_weights)                          # importance ratio
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)  # PPO clip

# 标准 "grpo" loss:
per_token_loss1 = coef_1 * advantages
per_token_loss2 = coef_2 * advantages
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```

- `epsilon_low / epsilon_high` = PPO 的 ε（默认 0.2/0.2）。
- `old_per_token_logps` = **生成那一刻**的 logp；当 `num_iterations=1` 且 `steps_per_generation ≤ gradient_accumulation_steps` 时，policy 没动，`old == current`，可以省掉重算（L2740-2741 注释）。

最终归约（L2784-2786, `loss_type=="grpo"`）：

```python
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
```

（先按 completion 长度平均，再 batch 平均。）

### 2.7 KL 到 reference (L2790-2796, 2822)

```python
if self.beta != 0.0:
    ref_per_token_logps = inputs["ref_per_token_logps"]
    per_token_kl = (
        torch.exp(ref_per_token_logps - per_token_logps)
        - (ref_per_token_logps - per_token_logps) - 1
    )   # = Schulman k3 estimator: exp(Δ) - Δ - 1,  Δ = logπ_ref - logπ_θ
...
if self.beta != 0.0:
    per_token_loss = per_token_loss + self.beta * per_token_kl
```

- 这是 John Schulman blog 里的 **k3 KL estimator**（始终 ≥0，无偏近似）。对应 week13「KL approximator」段。
- **`beta=0` 时根本不加载 ref 模型** (L800-801)：`if self.beta == 0.0: self.ref_model = None`。这一行对 0.8B 显存预算至关重要。
- `ref_per_token_logps` 在 `_generate_and_score_completions` (L2338-2357) 里**每个 generation batch 现算一次**（不像 DPO 可以 dataset-level 预算），存进 `inputs["ref_per_token_logps"]`。
- Reference 来源跟 DPO 一样的三分支：`beta=0` → None；PEFT → disable adapter；否则 → `create_model_from_path` 全副本 (L816)。

---

## 3. DPO vs GRPO 工程差异（一句话对比）

| 维度 | DPOTrainer | GRPOTrainer |
|---|---|---|
| **数据** | 离线 chosen/rejected 对（数据集现成） | 只有 prompt，**自己生成 G 个 completion + 现算 reward** |
| **策略类型** | off-policy（用历史对） | on-policy（每步重新 rollout） |
| **Reward 来源** | 隐式：靠 chosen vs rejected 的偏好对 | 显式：`reward_funcs`（RM 或规则函数）打分 |
| **Advantage** | 无（直接用 log-ratio） | group-relative `(R−μ)/σ`，无 critic |
| **每步前向数** | policy chosen+reject + ref chosen+reject ≈ 4 次半 forward | 1 次生成 G 个 + 1 次 policy forward + 1 次 ref forward |
| **显存峰值** | policy + ref（除非 PEFT/预计算） | policy + ref + reward model(s) + KV cache（生成期） |
| **Surrogate** | 直接 `-log σ(β·Δ)`，无 clip | PPO clip `-min(r·A, clip(r)·A)` |
| **KL 项** | 隐式在 `log π − log π_ref` 里（β 乘 log-ratio） | 显式 per-token KL 加到 loss 上（β 乘 KL） |

---

## 4. TBD / 风险（0.8B on Apple Silicon）

1. **GRPO 的 generation 成本 = 主瓶颈**。每步要生成 `per_device_train_batch_size × G` 个 completion（默认 G=8 → 8 倍推理开销）。0.8B 在 MX GPU 上单条 ~256 token 大约 1-2s，batch=4×G=8 = 32 条 → 每步生成 ~30-60s，一个 epoch 几小时起。考虑先把 G 调到 4 验证管线。
2. **Reference model = 2× 参数**。0.8B fp16 ≈ 1.6GB，policy + ref = 3.2GB 权重，加激活/KV cache 在 MX (16/32GB) 上还行；但 QLoRA + PEFT 可以**完全省掉 ref 模型**（DPO 和 GRPO 都支持，靠 disable adapter），是 Apple Silicon 上的推荐路径。
3. **`beta=0` 跑 GRPO** 直接不加载 ref 模型 (L800-801)，省一半显存。代价：无 KL 约束，policy 容易漂移。短期实验可以试 `beta=0` + 较小 lr，长期建议 PEFT。
4. **`precompute_ref_log_probs=True`（仅 DPO）**：把 ref 跑一次写回 dataset，训练时只占 policy 显存。代价：ref 是固定的（不能 sync），且不支持 PEFT 路径之外的 config 改动。
5. **`selective_log_softmax`** 在 Apple Silicon 上是否走 Metal 还需 profile；若 fallback 到 CPU gather，长序列会慢。可考虑 `use_liger_kernel=True`（fused kernel），但 Liger 对 MPS 后端支持要查（DPO 的 `_compute_loss_liger` L1216 / GRPO 的 `compute_liger_loss` L2566）。
6. **没有发现的符号**（如实记录）：`dpo_loss`、`dp_loss`、`get_batch_logps`、`compute_logps` 这些老版 TRL 教程里的方法名 **在 main 分支已不存在**，所有逻辑进了 `_compute_loss`。如果跑通后看到 import error 引用这些，是版本差异。
7. **DPO 的 `[chosen, rejected]` batch 顺序是隐性 contract**，自定义 collator 必须遵守；GRPO 的 `view(-1, num_generations)` 同理——sampler 已经保证，但自定义 dataloader 要小心。
8. **DPO 多 loss 混权**（`loss_types` 是 list）方便做消融（如 `sigmoid + ipo` 加权），但要注意 IPO 用 length-normalized score，跟 sigmoid DPO 的 Σ logp 数值尺度差很大，混权时 `loss_weights` 要调。

---

**核心一句话**：DPO 是「数据驱动的偏好学习」——离线对 + 减 log-ratio + log-sigmoid；GRPO 是「on-policy + 自打分 + group-baseline 替代 critic」——生成 G 个 → group-mean/std 归一 reward → PPO clip → 加 KL。两者都需要 reference policy，但 TRL 给了 3 条省 ref 显存的路（PEFT/预计算/`beta=0`），Apple Silicon 0.8B 上首推 PEFT。
