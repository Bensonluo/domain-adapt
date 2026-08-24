# Week 29：开源发布准备

> 目标：把代码、文档、notebook 和复现流程整理到 v0.1 发布质量。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1

完成格式、类型、安全和依赖检查。

### Day 2–3

完成 Quick Start、模块文档、API 和复现指南。

### Day 4

完成三个示例 notebook。

### Day 5

生成 release note 和 v0.1 验收报告。

## 任务清单

- [ ] w29-1 完成代码质量和安全检查
- [ ] w29-2 完成文档与复现指南
- [ ] w29-3 添加三个示例 notebook
- [ ] w29-4 准备 AdaptStack v0.1

## 执行

```bash
bash phase2/week29/run_week29.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week29/validate_week29.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week29/validate_week29.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week29/tests -v
```

## 交付物

- `adaptstack/docs/quickstart.md`
- `adaptstack/docs/reproduction.md`
- `adaptstack/notebooks/cpt.ipynb`
- `adaptstack/notebooks/sft_grpo.ipynb`
- `adaptstack/notebooks/distillation.ipynb`
- `results/week29_release/release_checklist.md`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
