# Week 32：Phase 2 复盘 + 成果沉淀

> 目标：收口代码、论文、Demo、证据和可复用方法论。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

整理交付物和跨领域通用性证据。

### Day 3

总结医疗验证边界、失败和后续工作。

### Day 4

准备 30 分钟深度技术分享。

### Day 5

完成 Phase 2 全面复盘和最终验收。

## 任务清单

- [ ] w32-1 沉淀全部成果档案
- [ ] w32-2 总结通用性和医疗边界
- [ ] w32-3 准备 30 分钟技术分享
- [ ] w32-4 完成 Phase 2 复盘

## 执行

```bash
bash phase2/week32/run_week32.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week32/validate_week32.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week32/validate_week32.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week32/tests -v
```

## 交付物

- `results/week32_retrospective/deliverables.md`
- `results/week32_retrospective/domain_boundary.md`
- `results/week32_retrospective/talk_outline.md`
- `results/week32_retrospective/phase2_summary.md`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
