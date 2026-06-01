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

- [ ] 医疗语料数据集（1B+ tokens）
- [ ] 3 种混合比例的配置文件
- [ ] Token 统计报告

---

## 验收清单

- [ ] 医疗语料清洗完成，1B+ tokens
- [ ] 4 种混合比例（纯领域 / 70-30 / 50-50 / 30-70）配置就绪
- [ ] Token 统计报告生成
