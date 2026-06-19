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

- [x] 真实 CPT 语料已接通（魔搭：医学百科 905 万 + 中文维基 509 万 token，4 配比就绪）— [`data/processed/cpt_ready/cpt_*.jsonl`](../data/processed/cpt_ready/)（demo synthetic 数据已被真实数据替换）
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

> **Tokenizer 与 week11 的关系**：week10 的 `--tokenizer` 生成 `ids` 字段 + 统计 token 数。
> week11 的 MLX 训练**只用 `text` 字段**，用目标模型（Qwen3.5-0.8B-Base）自己的 tokenizer 在线重编码——
> 所以 tokenizer 选错**不影响训练正确性**。但 ⚠️ `token_report.json` 的规模数字必须用**目标模型同源 tokenizer** 才准:
> 实测同一句中文「本章介绍了临床药理学的概念与发展概况。」fake(字节级)= 57 token，Qwen3.5 BPE = 9 token（**压缩 6.3x**）。
> 跨代 tokenizer（Qwen2.5→3.5）词表也不同——用错则统计的 token 数差几倍，步数/显存预算全错。

**真实数据（week12 已接通）**：用魔搭 ModelScope 真实中文语料替换合成数据，让 CPT 真正能学到领域知识
（week11 已证：16 条 fake-tokenizer 合成数据只学「医疗腔调」form，学不到 fact）。

| 源 | 内容 | 条数 | tokens |
|----|------|------|--------|
| 领域 `zjydiary/Medical` → `pretrain/medical_book_zh.json` | 中文医学百科纯文本 | 7,610 | 905 万 |
| 通用 `AI-ModelScope/wikipedia-cn-20230720-filtered` | 中文维基纯文本 | 10,000 | 509 万 |

```bash
# 1. 下载真实语料（魔搭，国内友好；脚本见 download_corpus.py）
python phase1/week10/download_corpus.py        # 领域+通用各取上限 → phase1/data/raw/{domain,general}/

# 2. 用真实 Qwen3.5-0.8B-Base tokenizer 重跑 4 配比
python phase1/week10/data_prep_cpt.py --source local --corpus phase1/data/raw \
    --tokenizer models/Qwen3.5-0.8B-Base-ms
```

产出（vs week11 假数据 16 条 / 3.3 万 token）：4 配比 chunk 数 70-30=6315 / 100-0=4420 / 50-50=4971 / 30-70=3551，
token 比精确命中目标（实测 70-30 实际 0.70 / 目标 0.70），**真实 token 量提升 ~200 倍**（905 万医疗 + 509 万维基）。
重训验证（mlx venv）留 week13。
