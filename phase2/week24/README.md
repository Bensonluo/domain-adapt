# Week 24：推理层 + 评估层串联

> 目标：完成 vLLM、量化、RAG 和三层评估，并跑通第一条完整 AdaptStack 训练。
> 状态：脚手架已就绪，业务实现待本周完成

### Day 1–2

实现 vLLM serving、INT8/INT4 量化和统一推理接口。

### Day 3

实现向量检索与 rerank 的 RAG 模块。

### Day 4–5

串联 benchmark、LLM judge、人工评估并完成首次全流程训练。

## 任务清单

- [ ] w24-1 实现 vLLM 和量化
- [ ] w24-2 实现 RAG
- [ ] w24-3 实现三层评估工具
- [ ] w24-4 跑通第一次完整训练

## 执行

```bash
bash phase2/week24/run_week24.sh
```

运行前只检查本周脚手架：

```bash
phase1/.venv/bin/python phase2/week24/validate_week24.py --scope code
```

本周完成后执行完整验收：

```bash
phase1/.venv/bin/python phase2/week24/validate_week24.py --scope complete
phase1/.venv/bin/python -m unittest discover -s phase2/week24/tests -v
```

## 交付物

- `adaptstack/src/adaptstack/inference/vllm_deploy/runner.py`
- `adaptstack/src/adaptstack/inference/rag/pipeline.py`
- `adaptstack/src/adaptstack/eval/pipeline.py`
- `results/week24_first_run/week24_summary.json`

## 自测题

1. 本周每个任务对应的输入、输出和验收指标是什么？
2. 本周结果与上一周是否使用同一评估口径？
3. 失败结果、限制和不可复现因素是否被明确记录？

## 验收清单

- [ ] 运行脚本非零失败、不会打印伪成功
- [ ] 完整验证器通过
- [ ] 测试通过
- [ ] 真实交付物存在且可追溯
