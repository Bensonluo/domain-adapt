# Week 7: 数学推导密集周

> 目标: 5 个核心推导,每个都从"已知"出发,一步步走到"结论"。
> 预计时间: 10-14 小时

> **上周回顾**: Week 6 你训练了完整的领域模型 — 全是工程实践。这周回到数学,因为工程做到一定程度后,瓶颈往往是"直觉不够"。推导的目的不是记住公式,而是建立直觉。
>
> **为什么学这周**: 这 5 个推导覆盖了 LLM 训练的核心数学。你不需要记住每一步,但你需要理解"结果为什么长这样"。比如 DPO loss 的直觉是 "让好回答的概率相对变高,坏回答的概率相对变低" — 推导只是把这句话变成公式。
>
> **怎么用这周**: 每个 derivation 模板文件 (`derivation_*.md`) 包含: **起点**(已知什么)、**终点**(要证什么)、**步骤**(中间路径)、**检查点**(你到这一步时应该得到的结果)。如果你卡住了,看下一步的第一行提示,而不是整个答案。

---

## 推导 1: Self-Attention 反向传播 (Day 1)

### 起点 → 终点
```
已知: S = QK^T / √d_k,  O = softmax(S) @ V
求:   ∂L/∂Q,  ∂L/∂K,  ∂L/∂V
```

### 步骤 (跟着走,每步算一个中间结果)

**Step 1: 求 ∂L/∂V** (最简单,先热身)
- 已知 `O = softmax(S) @ V`
- 这是矩阵乘法,∂L/∂V = ?
- **检查点**: 应该得到一个包含 softmax(S) 的表达式

**Step 2: 求 ∂L/∂S** (中间量,后续需要)
- 已知 `O = softmax(S) @ V`,并且你已经有 ∂L/∂O
- V^T 在哪边? ∂L/∂S = ∂L/∂O @ V^T × ∂softmax/∂S
- softmax 的 Jacobian: `∂softmax(s_i)/∂s_j = p_i(δ_ij - p_j)`
- **检查点**: 结果应该包含 `P @ (something)` 其中 P = softmax(S)

**Step 3: 求 ∂L/∂Q 和 ∂L/∂K**
- 已知 `S = QK^T / √d_k`
- ∂L/∂Q = ∂L/∂S × ∂S/∂Q
- 矩阵乘法的梯度: `∂(AB)/∂A` 的规则是什么?
- **检查点**: ∂L/∂Q 应该和 K 有关,∂L/∂K 应该和 Q 有关

### 梯度爆炸直觉
> 当 softmax 接近 one-hot (某个 logit 远大于其他),梯度趋近 0。这就是为什么需要 scaling (÷√d_k) — 防止点积太大导致 softmax 尖锐化。

### 交付物
- [ ] 手写推导照片 → `phase0/notes/week7_derivation_attention.jpg`
- [ ] 能在白板上解释 "attention 梯度为什么可能爆炸"

**参考**:
- https://medium.com/@dzqueque/deriving-the-self-attention-gradient-formula
- Raschka LLMs-from-scratch Ch3

---

## 推导 2: Softmax + Cross-Entropy 梯度 (Day 2)

### 起点 → 终点
```
已知: L = -log(softmax(z)_y),  softmax(z)_i = e^z_i / Σe^z_j
求:   ∂L/∂z_i = ?
```
预期结果: `p_i - 1(i=y)` (概率减去 one-hot)

### 步骤

**Step 1: 拆分**
- `L = -log(p_y)` 其中 `p_y = softmax(z)_y`
- 用链式法则: `∂L/∂z_i = ∂L/∂p_y × ∂p_y/∂z_i`

**Step 2: 第一项**
- `∂L/∂p_y = -1/p_y`

**Step 3: 第二项 (分两种情况)**
- 情况 i = y: `∂p_y/∂z_y = p_y(1 - p_y)`
- 情况 i ≠ y: `∂p_y/∂z_i = -p_y × p_i`
- **检查点**: 这两个加起来应该得到 `p_y(δ_{iy} - p_i)`

**Step 4: 合并**
- `∂L/∂z_i = -1/p_y × p_y(δ_{iy} - p_i) = -(δ_{iy} - p_i) = p_i - δ_{iy}`
- **这就是最终答案!**

### 数值稳定性补充
> 实际代码不用 `log(softmax(z))` — 用 `log_softmax(z)`,它内部先减 max(z) 再算,避免数值溢出。推导时假设数值稳定,结论不变。

### 交付物
- [ ] 手写推导照片 → `phase0/notes/week7_derivation_softmax_ce.jpg`
- [ ] 验收: 5 分钟内在白板上推完

---

## 推导 3: LoRA 的 SVD 视角 (Day 3)

### 起点 → 终点
```
已知: ΔW = B @ A,  B∈R^{d×r},  A∈R^{r×k},  rank(ΔW) ≤ r
理解: 为什么 "低秩" 是合理的? rank 选多少够?
```

### 步骤

**Step 1: 理解 "低秩" 的含义**
- 矩阵的 rank = 独立行/列的数量
- ΔW = B @ A,rank 最多 = r (因为中间维度是 r)
- 直觉: ΔW 的所有列都是 A 的行的线性组合 → 只有 r 个自由度

**Step 2: SVD 分解的连接**
- 任何矩阵 W 都可以写成: `W = UΣV^T = Σ σ_i u_i v_i^T`
- 前 r 个奇异值捕获的能量 = `Σ_{i=1}^{r} σ_i^2 / Σ σ_i^2`
- 打开 `lora.ipynb` 看你 Week 4 画的奇异值衰减曲线

**Step 3: LoRA 为什么有效**
- 关键假设: 预训练权重 W_0 已经编码了大部分知识
- 微调只需要一个 "小修正" ΔW
- 如果 W_0 的奇异值衰减快 → 低秩修正就够了
- **检查点**: 在 lora.ipynb 中,rank=32 捕获了多少能量? 这和论文用 r=8 不矛盾吗? (提示: ΔW 不是 W,是修正量)

**Step 4: alpha 的作用**
- `scaling = alpha / rank`
- 为什么不直接设 alpha = rank (scaling = 1)?
- 好处: 调 rank 时不需要重新调 learning rate (alpha 固定)
- **直觉**: alpha 是 "LoRA 的总强度",rank 是 "LoRA 的自由度"

### 交付物
- [ ] 手写推导照片 → `phase0/notes/week7_derivation_lora_svd.jpg`
- [ ] 能回答 "rank=8 通常够用的数学直觉是什么"

---

## 推导 4: DPO Loss 推导 (Day 4)

> 这个推导最长,但思路清晰: 从 RLHF 的目标出发,经过 3 步变换,消掉不可计算的配分函数。

### 起点 → 终点
```
已知: RLHF 目标 max E_π[log π(y|x)] - β KL(π || π_ref)
求:   DPO 的闭式 loss
终点: L_DPO = -E[log σ(β (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))]
```

### 步骤

**Step 1: RLHF 目标 → Reward Model 参数化**
- RLHF 优化: `max_π E[reward(x,y)] - β KL(π || π_ref)`
- 最优解: `π*(y|x) = (1/Z(x)) × π_ref(y|x) × exp(reward(x,y)/β)`
- 其中 `Z(x) = Σ_y π_ref(y|x) exp(reward(x,y)/β)` 是配分函数
- **检查点**: 到这里,你应该理解 "直接优化 π 需要知道 Z(x),但 Z(x) 不可计算"

**Step 2: 反解 reward**
- 从 Step 1 反解: `reward(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)`
- 关键观察: `Z(x)` 只依赖 x,不依赖 y
- 对于一对 (y_w, y_l): reward 差值中 Z(x) 会消掉!
- **检查点**: `reward(x,y_w) - reward(x,y_l) = β [log(π*(y_w)/π_ref(y_w)) - log(π*(y_l)/π_ref(y_l))]`

**Step 3: 代入 Bradley-Terry 模型**
- 偏好模型: `P(y_w > y_l) = σ(reward(x,y_w) - reward(x,y_l))`
- 把 Step 2 的结果代入: `P = σ(β [log π*(y_w)/π_ref(y_w) - log π*(y_l)/π_ref(y_l)])`
- 取 negative log-likelihood: 这就是 DPO loss!
- **最终检查**: 确认 Z(x) 完全消失了 ✓

### 直觉总结
> DPO loss 的含义: "让好回答相对 π_ref 的概率升高,让坏回答相对 π_ref 的概率降低"。β 控制强度。
> 关键技巧: 利用 Bradley-Terry 的成对比较,消掉了不可计算的 Z(x)。

### 交付物
- [ ] 手写推导照片 → `phase0/notes/week7_derivation_dpo.jpg`
- [ ] 能用一句话解释 DPO 相对 RLHF 的优势

**参考**:
- DPO 论文: https://arxiv.org/abs/2305.18290
- https://huggingface.co/blog/pref-tuning

---

## 推导 5: AdamW 更新规则 (Day 5)

### 起点 → 终点
```
已知: Adam 的更新规则
求:   AdamW 做了什么修改,为什么 L2 正则不够好
```

### 步骤

**Step 1: 写出 Adam 的完整公式**
```
m_t = β1 × m_{t-1} + (1-β1) × g_t         # 一阶矩 (动量)
v_t = β2 × v_{t-1} + (1-β2) × g_t^2       # 二阶矩 (自适应学习率)
m̂_t = m_t / (1 - β1^t)                     # 偏差修正
v̂_t = v_t / (1 - β2^t)                     # 偏差修正
θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)
```

**Step 2: L2 正则 (Adam + weight decay)**
- 在梯度上加 λθ: `g_t = ∂L/∂θ + λθ`
- 问题: 这个 λθ 会经过 m 和 v 的缩放
- 实际 decay 量 = `lr × λθ × (自适应缩放因子)`
- 自适应缩放因子对不同参数不同 → weight decay 效果不均匀
- **检查点**: 你应该理解 "为什么经过 m/v 之后 decay 不均匀了"

**Step 3: AdamW 的解耦**
- 把 weight decay 从梯度里拿出来,直接作用在参数上:
- `θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε) - lr × λ × θ_{t-1}`
- weight decay 不经过 m/v,是常数比例的直接衰减
- **这就是 AdamW 和 Adam+L2 的唯一区别**

**Step 4: 为什么 LLM 用 AdamW**
- 大模型参数多,不同参数的梯度量级差异大
- Adam 的自适应学习率让每个参数有独立的更新步长
- 如果 weight decay 也被自适应缩放 → 大梯度的参数 decay 多,小梯度的 decay 少 → 不合理
- AdamW 解耦后,所有参数的 decay 比例一致 → 更合理

### 交付物
- [ ] 手写推导照片 → `phase0/notes/week7_derivation_adamw.jpg`
- [ ] 能在一句话内说清 AdamW vs Adam+L2 的区别

---

## Day 6-7: 复习 + 模拟白板

### 复习方法 (间隔重复)
1. **第一遍** (Day 6): 不看笔记,重新推 5 个,限时 15 分钟每个
2. **标记卡顿点**: 哪一步需要看提示? 标记出来
3. **第二遍** (Day 7): 只推卡顿的部分,每个 5 分钟
4. **白板模拟**: 找一面墙,讲给空气听,录音回听

### 交付物
- [ ] 5 份手写推导 (拍照存档)
- [ ] 2 次完整复习记录
- [ ] 1 次模拟白板录音

---

## 自测题

1. **Attention 反向传播中,如果不除以 √d_k,梯度会出现什么问题?**
2. **DPO 的关键数学技巧是什么?** (消掉了什么?)
3. **AdamW 把 weight decay 从梯度中解耦出来,为什么这比 Adam+L2 更好?**

> 答案: 1) Q·K 的方差随 d_k 增大而增大,softmax 趋近 one-hot,梯度趋近零,训练停滞。2) 利用 Bradley-Terry 成对偏好模型消掉了不可计算的配分函数 Z(x)。3) Adam+L2 的 weight decay 会经过 m/v 的自适应缩放,对不同参数 decay 不均匀;AdamW 的 decay 直接作用于参数,所有参数 decay 比例一致。

---

## 成果

**推导 1: Self-Attention 反向传播** — 从 O = softmax(QK^T/√d_k)V 出发，逐步求 ∂L/∂V、∂L/∂P、∂L/∂S、∂L/∂Q、∂L/∂K，解释 √d_k 缩放的数学依据（方差随 d_k 线性增长导致 softmax 尖锐化）。见 [derivation_attention.md](derivation_attention.md)。

**推导 2: Softmax + Cross-Entropy 梯度** — 用两种方法（直接展开法、链式法则法）推导出 ∂L/∂z = p - y_onehot，即"概率减 one-hot = 预测误差"。附带 log-sum-exp 数值稳定性分析。见 [derivation_softmax_ce.md](derivation_softmax_ce.md)。

**推导 3: LoRA 的 SVD 视角** — 从 SVD 分解和 Eckart-Young 定理出发，解释低秩假设的数学依据：预训练权重的奇异值衰减快，微调变化量 ΔW 的衰减更快，因此 r=8 通常足够。详解 alpha/rank 的解耦设计。见 [derivation_lora_svd.md](derivation_lora_svd.md)。

**推导 4: DPO Loss** — 从 RLHF 的 KL 约束优化目标出发，经闭式解、反解 reward、Bradley-Terry 模型三步，消掉不可计算的配分函数 Z(x)，得到 DPO 闭式 loss。含梯度分析和代码实现。见 [derivation_dpo.md](derivation_dpo.md)。

**推导 5: AdamW 更新规则** — 从 SGD 到 Adam（动量+自适应学习率），再到 Adam+L2 的缺陷（weight decay 被自适应缩放扭曲），最后到 AdamW 的解耦设计。含数值示例和 LLM 训练场景的分析。见 [derivation_adamw.md](derivation_adamw.md)。
