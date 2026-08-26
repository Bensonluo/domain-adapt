# Week 30：arXiv 技术报告写作

> 目标：完成有具体数字、honest limitations 和可追溯实验表格的技术报告。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1

Introduction + Related Work。

### Day 2

Methodology：四层架构和训练 pipeline。

### Day 3

Experiments：ablation 和跨域验证。

### Day 4–5

Analysis、Limitations、图表、引用和全文构建。

## 任务清单

- [ ] w30-1 写 Introduction + Related Work
- [ ] w30-2 写 Methodology
- [ ] w30-3 写 Experiments
- [ ] w30-4 写 Analysis + Limitations

## 执行

```bash
bash phase2/week30/run_week30.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week30/validate_week30.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week30/validate_week30.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week30/tests -v
```

## 交付物

- `paper/main.tex`
- `paper/references.bib`
- `paper/figures`
- `results/week30_paper/build_report.md`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
