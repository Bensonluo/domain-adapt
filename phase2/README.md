# Phase 2：AdaptStack 整合阶段

> Week 22–32 | AdaptStack 开源项目 + 系统实验 + Demo + 技术报告

本目录沿用 Phase 0/1 的组织方式：每一周是 `weekXX/` 同级目录，周内包含
README、运行脚本、验证器和测试；共享项目代码位于 `adaptstack/`，数据、结果、
笔记和公共工具位于 Phase 2 根目录。

## 目录

```text
phase2/
├── adaptstack/          # Phase 2 持续迭代的开源项目
├── data/                # Phase 2 输入与处理中间数据
├── notes/               # 调研、决策和复盘笔记
├── results/             # 按周保存真实实验结果
├── utils/               # 分周共享验证工具
├── week22/ ... week32/  # 与 phase0/phase1 一致的周目录
└── requirements.txt
```

## 分周计划

| 周 | 主题 | 当前状态 |
|---|---|---|
| Week 22 | AdaptStack 架构、接口、README 和项目骨架 | 已完成 |
| Week 23 | 数据层 + 训练层串联 | 待实现 |
| Week 24 | 推理层 + 评估层 + 首次完整训练 | 待实现 |
| Week 25 | Ablation 1–3 | 待实现 |
| Week 26 | Ablation 4–6 + 跨域验证 | 待实现 |
| Week 27 | 分析与 Insight 提炼 | 待实现 |
| Week 28 | Web Demo 开发和部署 | 待实现 |
| Week 29 | 开源发布准备 | 待实现 |
| Week 30 | arXiv 技术报告 | 待实现 |
| Week 31 | 推广与社区反馈 | 待实现 |
| Week 32 | Phase 2 复盘和成果沉淀 | 待实现 |

任务名称与 `portfolio-fe/data/growing-big.ts` 的 Phase 2 进度项对应；执行结构以
本仓库已有的 `phase0/weekX`、`phase1/weekX` 为准。

## 一步一步执行

```bash
# 查看当前周说明、运行与测试
sed -n '1,240p' phase2/week22/README.md
bash phase2/week22/run_week22.sh
phase1/.venv/bin/python phase2/week22/validate_week22.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week22/tests -v

# Week 23 的后续实现入口
sed -n '1,240p' phase2/week23/README.md
bash phase2/week23/run_week23.sh
```

尚未实现的周会返回退出码 3，提示该周尚未实现，不生成实验结果。
