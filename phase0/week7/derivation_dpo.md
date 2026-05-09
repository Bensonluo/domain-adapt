# 推导 4: DPO Loss 推导

## 起点: RLHF 目标

```
max E_π[log π(y|x)] - β KL(π || π_ref)
```

其中:
- π: 当前策略 (要优化的模型)
- π_ref: 参考策略 (SFT 模型)
- β: KL 惩罚系数

## Step 1: 通过 reward model 参数化

TODO: 推导
```
r(x,y) = β log(π(y|x) / π_ref(y|x)) + β log Z(x)
```

提示: 从 KL 约束的优化问题出发,用 Lagrange multiplier

## Step 2: Bradley-Terry 偏好模型

TODO: 推导
```
P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
```

其中 σ 是 sigmoid 函数

## Step 3: 消掉 Z(x)

TODO: 推导为什么 Z(x) 会被消掉

最终得到 DPO 闭式目标:
```
L_DPO = -E[log σ(β (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

## 关键理解

- DPO 的巧妙之处: **不需要显式训练 reward model**
- 偏好数据 (y_w, y_l) 直接约束策略
- β 控制 "偏离参考策略的程度"

---

**参考**:
- DPO 论文: https://arxiv.org/abs/2305.18290
- https://huggingface.co/blog/pref-tuning

**拍照存档**: `phase0/notes/week7_derivation_dpo.jpg`
