# Week 31：推广 + 社区互动

> 目标：发布项目、模型和数据集，系统记录真实社区反馈。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1

发布项目介绍 thread 和 Demo 链接。

### Day 2

发布 Hugging Face 模型卡和数据卡。

### Day 3

发布技术社区长文并整理反馈。

### Day 4–5

向目标工程师征求反馈，形成 issue backlog。

## 任务清单

- [ ] w31-1 发布项目 thread
- [ ] w31-2 发布 Hugging Face 模型和数据集
- [ ] w31-3 社区分享并记录反馈
- [ ] w31-4 定向征求工程师反馈

## 执行

```bash
bash phase2/week31/run_week31.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week31/validate_week31.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week31/validate_week31.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week31/tests -v
```

## 交付物

- `results/week31_community/release_links.md`
- `results/week31_community/model_card.md`
- `results/week31_community/dataset_card.md`
- `results/week31_community/feedback_summary.md`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
