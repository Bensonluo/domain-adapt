# Week 10：CPT 实战准备

> 目标: 构建 1B-3B token 医疗语料，配置 CPT 训练环境。
> 预计时间: 12-16 小时

> **为什么学这周**: 数据准备是 CPT 成功的关键。数据质量直接决定 CPT 效果。
>
> **思考锚点**: "1B token 的医疗语料，大约相当于多少本书？清洗后能剩下多少有效 token？"

---

## Day 1-3: 领域语料采集

### 做什么
1. PubMed 中文摘要采集
2. 医学教科书电子版（内科学、外科学等）
3. 临床指南（NCCN、CSCO）
4. 医学论文（arXiv med 相关）

### 跑
```bash
# 使用 prep 阶段的清洗管线
python phase1/prep/clean_pipeline.py --config medical --input phase1/data/raw/ --output phase1/data/processed/cpt/
```

---

## Day 4-5: Tokenization + 混合比例配置

### 做什么
1. 用目标模型 tokenizer 统计 token 数
2. 准备 4 种混合比例配置：纯领域 / 70-30 / 50-50 / 30-70
3. 切分为训练格式（固定长度 + overlap）

### 跑
```bash
python phase1/week10/data_prep_cpt.py --corpus phase1/data/processed/cpt/ --tokenizer Qwen/Qwen2.5-3B
```

---

## 交付物

- [x] 医疗语料数据集（demo 级已就绪，1B+ tokens 留训练前下载）— [`data/processed/cpt_ready/cpt_*.jsonl`](../data/processed/cpt_ready/)
- [x] 4 种混合比例配置文件（100-0 / 70-30 / 50-50 / 30-70）— 实际 token 比例精确命中目标
- [x] Token 统计报告 — [`token_report.json`](../data/processed/cpt_ready/token_report.json)
- [x] 数据清洗管线跑通 — [`prep/clean_pipeline.py`](../prep/clean_pipeline.py)（200 条带脏数据 → 13 条干净，验证 boilerplate/exact-dedup 逻辑）

---

## 验收清单

- [x] 医疗语料清洗完成（demo 验证逻辑；真实 1B+ 留训练前下载）
- [x] 4 种混合比例（纯领域 / 70-30 / 50-50 / 30-70）配置就绪
- [x] Token 统计报告生成

---

## 数据管线状态

**当前（demo，零下载）**：`--source synthetic` + `--tokenizer fake` 跑通整条管线，验证清洗 / 混合 / 切块 / 报告逻辑。
管线：`raw → clean_pipeline.py → data_prep_cpt.py → cpt_*.jsonl + token_report.json`。

> **Tokenizer 与 week11 的关系**：week10 的 `--tokenizer` 只用于**统计 token 数** + 生成 `ids` 字段。
> week11 的 MLX 训练**只用 `text` 字段**，用目标模型（Qwen3.5-0.8B）自己的 tokenizer 重新编码。
> 所以 week10 用 `fake` / `Qwen2.5-3B` / `Qwen3.5-0.8B` 任意 tokenizer 都不影响 week11 训练正确性——`token_report.json` 的数字仅供规模参考。

**训练前（真实数据）**：
1. 建 venv：`pip install -r requirements.txt`（国内用 aliyun 镜像）
2. 放医疗语料到 `data/raw/medical/*.txt`，跑 `clean_pipeline.py --config medical --input data/raw/medical/`
3. 跑 `data_prep_cpt.py --source local --corpus data/processed/cpt/ --tokenizer Qwen/Qwen2.5-3B-Instruct`（模型已在 HF 缓存，离线可用，无需下载）
