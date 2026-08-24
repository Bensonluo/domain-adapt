# Week 26：Ablation 实验（下）+ 跨域验证

> 目标：完成模型规模、RAG vs FT 和第二领域复现实验。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

实验 4：1.5B vs 3B。

### Day 3

实验 5：base+RAG vs SFT vs SFT+RAG。

### Day 4–5

实验 6：第二领域只替换数据和配置，复用同一 pipeline。

## 任务清单

- [ ] w26-1 模型规模实验
- [ ] w26-2 RAG vs FT 实验
- [ ] w26-3 医疗到第二领域跨域验证

## 执行

```bash
bash phase2/week26/run_week26.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week26/validate_week26.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week26/validate_week26.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week26/tests -v
```

## 交付物

- `experiments/week26_ablation.yaml`
- `configs/legal.yaml`
- `results/week26_cross_domain/week26_summary.json`
- `results/week26_cross_domain/statistics.json`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
