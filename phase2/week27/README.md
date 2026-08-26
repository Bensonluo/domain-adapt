# Week 27：分析 + Insight 提炼

> 目标：从完整实验中提炼可复核、能分层表达的独立结论。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

核对实验口径、显著性、失败结果和反例。

### Day 3–4

提炼 3–5 个 insight，并准备 30 秒/5 分钟讲法。

### Day 5

完成分析文档、限制和证据索引。

## 任务清单

- [ ] w27-1 提炼 3–5 个数据支持的 insight
- [ ] w27-2 准备分层讲法
- [ ] w27-3 写完整分析和局限

## 执行

```bash
bash phase2/week27/run_week27.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week27/validate_week27.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week27/validate_week27.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week27/tests -v
```

## 交付物

- `results/week27_insights/insights.md`
- `results/week27_insights/evidence_index.json`
- `results/week27_insights/limitations.md`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
