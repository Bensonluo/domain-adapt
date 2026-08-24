# Week 22：AdaptStack 项目架构设计

> 目标：把 Phase 0/1 的能力整理为 domain-agnostic 的四层项目骨架。
> 状态：已完成，可执行并验收

### Day 1–2

设计 data / training / inference / eval 四层架构和依赖方向。

### Day 3–4

定义模块接口、不可变运行上下文、配置和 stage registry。

### Day 5

完成英文 README、CONTRIBUTING、项目入口和测试。

## 任务清单

- [x] w22-1 设计 AdaptStack 4 层架构
- [x] w22-2 设计 domain-agnostic 模块接口
- [x] w22-3 完成英文 README 和 CONTRIBUTING.md
- [x] w22-4 初始化可执行项目骨架

## 执行

```bash
bash phase2/week22/run_week22.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week22/validate_week22.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week22/validate_week22.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week22/tests -v
```

## 交付物

- `adaptstack/docs/architecture.md`
- `adaptstack/docs/module-interfaces.md`
- `adaptstack/README.md`
- `adaptstack/CONTRIBUTING.md`
- `adaptstack/scripts/train.py`
- `adaptstack/tests/test_pipeline.py`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [x] 运行脚本非零失败、不会打印伪成功
- [x] 完整验证器通过
- [x] 测试通过
- [x] 真实交付物存在且可追溯
