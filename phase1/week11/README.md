# Week 11：CPT 实验（上）

> 方法学纠错见 [`CORRECTIONS.md`](CORRECTIONS.md)。本周 run 只作为 demo/链路证据；“全量”指 MLX full 模式和全部 transformer layers，不等同于全部模型参数可训练。

> 目标: 在 Qwen3.5-0.8B base model 上跑第一次 CPT，监控训练过程。
> 预计时间: 8-12 小时
> 框架: **Apple MLX (mlx-lm)** — 本次采用的 M3 Max 训练框架；与 transformers+torch 的性能差异需结合相同配置的实测比较。

> **思考锚点**: "CPT 的 loss 曲线和 SFT 的 loss 曲线有什么不同？为什么？"

---

## 为什么用 MLX 而不是 transformers

| 维度 | transformers + torch (mps) | mlx-lm (Apple 原生) |
|------|---------------------------|---------------------|
| 后端 | MPS 兼容层 | Apple Silicon 原生 (Unified Memory) |
| 0.8B 全量 CPT | 当前材料未提供与 MLX 同条件的性能结果 | 参考配置为 batch 4；本次已运行 demo 的 batch 为 1 |
| 数据格式 | `datasets`；文本格式由训练目标确定 | 本次采用含纯 `text` 字段的 `train.jsonl` |
| CLI | 通过 HF Trainer 配置训练 | 通过 `python -m mlx_lm lora` 启动训练 |

> 本次 CPT 使用纯 `text` 字段和全序列 loss，与所用 mlx-lm 数据接口对应；SFT 的 completion-only 目标另作比较。
>
> 参考配置与已运行配置分开记录：下方示例使用 batch 4，实际 demo 的 [`run_config.json`](../results/week11_cpt_pure/run_config.json) 记录 batch_size=1、15 条训练样本和 1 条验证样本。框架选择已完成，但这次运行不是两套框架的同条件性能对比。

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
# MLX full 模式 CPT demo（默认 70-30 配比, 200 iters 验证流程）:
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
learning_rate: 1.0e-5             # 本次保守起点；不作为通用倍率规律
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

- [x] `results/week11_cpt_pure/loss_log.csv` — 每步 loss（iter, kind, loss）
- [x] `results/week11_cpt_pure/loss_curve.png` — loss 曲线图
- [x] `results/week11_cpt_pure/adapters/` — 训练权重（iter 100 / 200 / final）
- [x] `results/week11_cpt_pure/generation_test.txt` — 生成测试（观察领域语言能力变化）
- [x] `results/week11_cpt_pure/run_config.json` — 可复现配置

---

## 验收清单

- [x] CPT 训练完成，loss 稳定下降 — train `1.29→0.012` 单调降；val 在 iter 50 见底后反弹（过拟合，见下方结论）
- [x] 验证 perplexity 低于基线 — 基线 `e^1.61≈5.0`，最低(iter 50)`e^0.37≈1.45`，终点(iter 200)`e^0.66≈1.93`，均低于基线
- [x] Loss 曲线截图保存 — `loss_curve.png`
- [x] 生成测试能看出医疗领域语言变化 — 学会医疗话语结构（腔调），但内容编造（见结论：form vs fact）

---

## 实验结论

> 配置（详见 `run_config.json`）：Qwen3.5-0.8B-Base 全量 CPT（`full` / `num_layers -1`，解冻 66% ≈ 498M/752M），70-30 demo 数据（16 条），200 iters，**batch_size 1**、lr 1e-5、seq 2048、grad_checkpoint。

### 1. loss 曲线：train / val 在 iter 50 分叉 = 过拟合

| iter | train loss | val loss |
|------|-----------|----------|
| 1    | —         | 1.612    |
| 10   | 1.288     | —        |
| 50   | 0.280     | **0.369（最低）** |
| 100  | 0.151     | 0.428    |
| 150  | 0.061     | 0.570    |
| 200  | **0.012** | 0.657    |

train 一路降到几乎 0，但 val 在 iter 50 触底后**反弹爬升**。train/val 分叉是 CPT 监控的核心信号：**早停点在 iter 50**，再训就是死记训练集、泛化变差。

### 2. 生成测试：学到"腔调"，没学到"知识"

prompt「患者男性，54 岁，主诉胸痛」→ 模型吐出「42 岁持续高热…腹痛…心力衰竭…头孢曲松治心梗…CSCO 指南」。

- **学到的**：医疗话语结构（查体 / 初步诊断 / 鉴别诊断 / 治疗 / 指南）——领域的 **form（分布、句式）**。
- **没学到的**：年龄、诊断、用药全对不上 prompt——领域的 **fact（事实知识）**。

### 3. 三个 insight

1. **CPT 改 form，不改 fact**。模型学会"用医疗的腔调说话"，但教不会"说对的医疗话"——后者要靠**真实、足量、准确**的数据。16 条 fake-tokenizer demo 数据只能验证流程、传不了知识。
2. **小数据下该 MLX full-mode demo 的验证 loss 很快见底**。但验证集仅 1 条，不能稳定估计“过拟合速度”；它只提示正式实验需要更大验证集、早停和重复运行。
3. **CPT loss 曲线的读法**（呼应思考锚点）：CPT 对纯 text 的**每个 token** 算 loss，曲线反映"领域分布的拟合程度"；过拟合的标志不是 loss 高，而是 **train 继续降、val 反弹**的分叉——这是 CPT（续训全 token）区别于 SFT（只算 completion）的监控要点。

---

## 常见问题

**Q: 内存不够（OOM）？**
A: 降 `--batch-size 2`、加 `--grad-accum 4`、确认 `--grad-checkpoint` 开着；或改 `--fine-tune-type lora`（LoRA-CPT 省很多）。

**Q: 想跑更大模型（3B+）？**
A: 全量 CPT 跑不动，走 LoRA-CPT + 量化模型 `mlx-community/Qwen3.5-3B-4bit`。

**Q: `num_layers` 默认 16 是什么意思？**
A: mlx-lm 的 `full` 模式下，`num_layers=-1` 会解冻全部 transformer layers；当前 run 记录的 trainable 参数约 66%，因此不能称全部模型参数都参与训练。正式命名以实际 trainable parameter 清单为准。

**Q: demo 数据的 text 开头怎么有乱码（`��素为主`）？**
A: week10 用 FakeTokenizer（UTF-8 字节级）切 chunk 时，chunk 边界可能切在多字节中文字符中间，`decode` 出来开头出现替换字符。**这只影响 demo 数据**（验证流程用）。真实训练前用真实 Qwen tokenizer 重跑 week10（`--tokenizer Qwen/Qwen3.5-0.8B`），BPE 的 decode 对任意 token 子集都合法，text 会正常。
