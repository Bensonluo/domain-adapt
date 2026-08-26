# Week 25：Ablation 实验（上）

> 目标：完成 CPT、对齐方法和真实/蒸馏数据三组严格对照。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

实验 1：直接 SFT vs CPT+SFT。

### Day 3

实验 2：SFT、DPO、GRPO、DPO+GRPO。

### Day 4–5

实验 3：真实、蒸馏、混合数据；统一汇总统计。

## 任务清单

- [ ] w25-1 CPT 必要性实验
- [ ] w25-2 DPO/GRPO 对齐方法实验
- [ ] w25-3 真实/蒸馏/混合数据实验

## 执行

```bash
bash phase2/week25/run_week25.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week25/validate_week25.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week25/validate_week25.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week25/tests -v
```

## 交付物

- `experiments/week25_ablation.yaml`
- `results/week25_ablation/week25_summary.json`
- `results/week25_ablation/statistics.json`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
