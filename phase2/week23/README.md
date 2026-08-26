# Week 23：数据层 + 训练层串联

> 目标：实现 raw → clean → tokenized → dataloaders，并串联 CPT → SFT → DPO/GRPO。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

实现数据读取、清洗、去重、tokenize 和数据集切分。

### Day 3–4

实现 CPT、SFT、DPO/GRPO 可跳过的训练 stage。

### Day 5

补齐 YAML 配置、WandB/本地追踪、smoke test 和端到端测试。

## 任务清单

- [ ] w23-1 实现数据 pipeline
- [ ] w23-2 实现训练 pipeline
- [ ] w23-3 YAML 驱动全部步骤
- [ ] w23-4 集成实验追踪

## 执行

```bash
bash phase2/week23/run_week23.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week23/validate_week23.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week23/validate_week23.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week23/tests -v
```

## 交付物

- `adaptstack/src/adaptstack/data/pipeline.py`
- `adaptstack/src/adaptstack/training/pipeline.py`
- `adaptstack/configs/medical.yaml`
- `results/week23_pipeline/week23_summary.json`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
