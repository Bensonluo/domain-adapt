# Week 28：Web Demo 开发

> 目标：实现可切换 base/SFT/DPO/GRPO/RAG 的回答对比 Demo 并部署。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

实现统一推理后端和多配置对比 API。

### Day 3–4

实现交互界面、指标展示、错误和安全边界。

### Day 5

Docker + Nginx 部署并完成 smoke test。

## 任务清单

- [ ] w28-1 开发多配置回答对比 Demo
- [ ] w28-2 支持训练/RAG 配置切换
- [ ] w28-3 Docker + Nginx 部署

## 执行

```bash
bash phase2/week28/run_week28.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week28/validate_week28.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week28/validate_week28.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week28/tests -v
```

## 交付物

- `adaptstack/demo/app.py`
- `adaptstack/demo/Dockerfile`
- `adaptstack/demo/nginx.conf`
- `results/week28_demo/deployment_report.md`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
