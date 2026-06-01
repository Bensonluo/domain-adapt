# Week 20：Feature Distillation + On-Policy 实验（主攻深度方向）

> 目标: 尝试 Feature-level 和 On-policy 蒸馏，三种策略完整对比。
> 预计时间: 16-24 小时

> **这是 Phase 1 的两个主攻深度方向之一。按深度执行方法论跑 3+ 变体。**
>
> **思考锚点**: "三种蒸馏方法的效果差距有多大？额外的成本值得吗？"

---

## Day 1-3: Feature Distillation

### 做什么
1. 加载 teacher（大模型）和 student（小模型）
2. 实现 Feature distillation loss（对齐中间层表示）
3. 训练 + 评估

### 核心实现
```python
class FeatureDistillationLoss(nn.Module):
    def __init__(self, teacher_hidden_dim, student_hidden_dim):
        super().__init__()
        self.projection = nn.Linear(student_hidden_dim, teacher_hidden_dim)

    def forward(self, teacher_hidden, student_hidden):
        projected_student = self.projection(student_hidden)
        return F.mse_loss(projected_student, teacher_hidden.detach())
```

### 跑
```bash
python phase1/week20/distill_feature.py \
    --teacher Qwen/Qwen2.5-14B \
    --student Qwen/Qwen2.5-3B \
    --output phase1/results/week20_distill_feature/
```

---

## Day 4-5: On-Policy Distillation

### 做什么
1. 小模型生成回答
2. GPT-4o 评分（1-5 分）
3. 用评分构造偏好对，做 DPO

### 跑
```bash
python phase1/week20/distill_on_policy.py \
    --student phase1/results/week11_cpt_pure/ \
    --judge gpt-4o \
    --output phase1/results/week20_distill_on_policy/
```

---

## 交付物

- [ ] Feature distillation 训练记录
- [ ] On-policy distillation 训练记录
- [ ] `results/week20_distillation_comparison.md` — 三种策略完整对比

---

## 验收清单

- [ ] Feature distillation 实验完成
- [ ] On-policy distillation 实验完成
- [ ] 三种蒸馏方法有完整对比数据
- [ ] 至少 1 个关于蒸馏效率 vs 效果的 trade-off 发现
