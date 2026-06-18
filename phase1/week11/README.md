# Week 11：CPT 实验（上）

> 目标: 在 Qwen3.5-0.8B base model 上跑第一次 CPT，监控训练过程。
> 预计时间: 8-12 小时
> 框架: **Apple MLX (mlx-lm)** — 适配 M3 Max，比 transformers+torch 在 Mac 上快很多、内存省

> **思考锚点**: "CPT 的 loss 曲线和 SFT 的 loss 曲线有什么不同？为什么？"

---

## 为什么用 MLX 而不是 transformers

| 维度 | transformers + torch (mps) | mlx-lm (Apple 原生) |
|------|---------------------------|---------------------|
| 后端 | MPS 兼容层 | Apple Silicon 原生 (Unified Memory) |
| 0.8B 全量 CPT | 勉强跑，慢 | 轻松，batch 4 跑得动 |
| 数据格式 | `datasets` + chat template | `train.jsonl` 纯 text（CPT 本就该如此） |
| CLI | HF Trainer 复杂配置 | `python -m mlx_lm lora` 一行起训 |

> CPT 用纯 `text` 字段、不套 chat template、每个 token 都算 loss（区别于 SFT 的 completion-only）——这正好和 mlx-lm 的原生数据格式对齐。

---

## Day 1-2: 环境 + 启动训练

### 0. 环境（训练前一次性，注意国内镜像）

```bash
# 必须先建 venv（不污染全局）
python3 -m venv .venv && source .venv/bin/activate

# 国内镜像装 mlx-lm（M3 Max arm64）
pip install -i https://mirrors.aliyun.com/pypi/simple/ "mlx-lm[train]"
pip install -i https://mirrors.aliyun.com/pypi/simple/ matplotlib   # 可选: loss 曲线
pip install -i https://mirrors.aliyun.com/pypi/simple/ swanlab      # 可选: web 实时曲线
```

### 1. 数据（来自 week10）

```bash
python phase1/week10/data_prep_cpt.py   # 产出 cpt_70-30.jsonl 等
```

### 2. 训练

```bash
# 基础全量 CPT（默认 70-30 配比, 200 iters 验证流程）:
python phase1/week11/train_cpt.py

# 实时 matplotlib 窗口（本地零登录, 推荐）:
python phase1/week11/train_cpt.py --live-plot

# SwanLab web 曲线（国内友好, 最专业）:
python phase1/week11/train_cpt.py --report-to swanlab

# LoRA-CPT（省内存, 改 fine-tune-type）:
python phase1/week11/train_cpt.py --fine-tune-type lora

# 真实训练（调大 iters）:
python phase1/week11/train_cpt.py --iters 10000 --batch-size 4 --grad-accum 4
```

### 训练配置参考

字段名 100% 来自 mlx-lm 源码（`mlx_lm/lora.py` 的 `CONFIG_DEFAULTS`），完整版见 [cpt_config.yaml](cpt_config.yaml)：

```yaml
model: Qwen/Qwen3.5-0.8B          # 全量 CPT 必须非量化原版
fine_tune_type: full              # full=全量CPT / lora=LoRA-CPT
num_layers: -1                    # -1=解冻所有层（真全量）; 默认16只解冻最后16层
iters: 200                        # demo 200; 真实训练 1万+
learning_rate: 1.0e-5             # CPT 比预训练小 10-100 倍
batch_size: 4
max_seq_length: 2048              # 匹配 week10 chunk
grad_checkpoint: true             # 省内存（慢约 20-30%）
steps_per_report: 10              # 每 10 步打印 loss（实时曲线数据源）
steps_per_eval: 50                # 每 50 步算 val loss + ppl
```

> 也可不经 train_cpt.py，直接用 mlx-lm CLI：`python -m mlx_lm lora --config phase1/week11/cpt_config.yaml`

---

## Day 3-5: 监控 + 分析

### 实时看 loss 曲线（三种方式，任选）

| 方式 | 命令 | 特点 |
|------|------|------|
| **csv + png**（默认） | （无需额外参数） | 实时写 `loss_log.csv` + 训练后画 `loss_curve.png`，零外部依赖 |
| **matplotlib 实时窗口** | `--live-plot` | 本地窗口实时刷新曲线，零登录 |
| **SwanLab web** | `--report-to swanlab` | 国内友好的 web dashboard，最专业，可分享 |

不论哪种，mlx-lm 都会每 `steps_per_report` 步往终端打印一次 `Iter N: Train loss X.XXX`（默认 10 步）。

### 交付物

- [ ] `results/week11_cpt_pure/loss_log.csv` — 每步 loss（iter, kind, loss）
- [ ] `results/week11_cpt_pure/loss_curve.png` — loss 曲线图
- [ ] `results/week11_cpt_pure/adapters/` — 训练权重
- [ ] `results/week11_cpt_pure/generation_test.txt` — 生成测试（观察领域语言能力变化）
- [ ] `results/week11_cpt_pure/run_config.json` — 可复现配置

---

## 验收清单

- [ ] CPT 训练完成，loss 稳定下降
- [ ] 验证 perplexity 低于基线（val loss 在 `loss_log.csv` 里，ppl = exp(loss)）
- [ ] Loss 曲线截图保存（`loss_curve.png`）
- [ ] 生成测试能看出医疗领域语言变化

---

## 常见问题

**Q: 内存不够（OOM）？**
A: 降 `--batch-size 2`、加 `--grad-accum 4`、确认 `--grad-checkpoint` 开着；或改 `--fine-tune-type lora`（LoRA-CPT 省很多）。

**Q: 想跑更大模型（3B+）？**
A: 全量 CPT 跑不动，走 LoRA-CPT + 量化模型 `mlx-community/Qwen3.5-3B-4bit`。

**Q: `num_layers` 默认 16 是什么意思？**
A: mlx-lm 的 `full` 模式下，`num_layers` 控制"解冻最后 N 层"，默认 16。要真正的全量 CPT 必须设 `-1`（train_cpt.py 默认就是 -1）。

**Q: demo 数据的 text 开头怎么有乱码（`��素为主`）？**
A: week10 用 FakeTokenizer（UTF-8 字节级）切 chunk 时，chunk 边界可能切在多字节中文字符中间，`decode` 出来开头出现替换字符。**这只影响 demo 数据**（验证流程用）。真实训练前用真实 Qwen tokenizer 重跑 week10（`--tokenizer Qwen/Qwen3.5-0.8B`），BPE 的 decode 对任意 token 子集都合法，text 会正常。
