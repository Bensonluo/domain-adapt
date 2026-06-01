# Week 8.5：数据工程准备（Phase 1 前置）

> 目标: 在进入 CPT/DPO 之前，准备好三类高质量数据集。
> 预计时间: 14-20 小时

> **为什么先做数据**: 后训练 70% 的工作量在数据工程上。数据质量决定实验结论的可信度。Garbage in, garbage out。

---

## 任务 A：领域语料采集 + 清洗 Pipeline

### 做什么
1. 采集领域数据源（医疗/金融可插拔）
2. 实现通用清洗 Pipeline（去重、过滤、质量分级）
3. 在医疗数据上跑通，生成清洗报告

### 跑
```bash
python phase1/prep/clean_pipeline.py --config medical
```

### 交付物
- [ ] `clean_pipeline.py` — 通用清洗管线代码
- [ ] `results/prep_cleaning_report.json` — 清洗统计报告
- [ ] 清洗后语料保存到 `data/processed/`

---

## 任务 B：三类训练数据集构建

### 数据集 1：CPT 语料
- [ ] 清洗后领域语料 tokenization
- [ ] 准备 3 种混合比例：纯领域 / 70-30 / 50-50
- [ ] 统计每种配置的 token 总数

### 数据集 2：SFT 指令数据
- [ ] 从公开数据集筛选 + 格式统一
- [ ] 构建质量分级（A/B/C 三档）
- [ ] 划分 train/test（90/10）

### 数据集 3：偏好数据
- [ ] 实现偏好数据生成 Pipeline（teacher vs student）
- [ ] 生成 2000+ 对偏好数据
- [ ] 长度偏差分析 + 人工抽检 100 对

### 跑
```bash
python phase1/prep/build_sft_data.py --input data/raw/ --output data/processed/sft/
python phase1/prep/build_preference_data.py --questions data/raw/questions.jsonl --n 2000
python phase1/prep/validate_data.py --dir data/processed/
```

### 交付物
- [ ] `build_sft_data.py` — SFT 数据构建脚本
- [ ] `build_preference_data.py` — 偏好数据构建脚本
- [ ] `validate_data.py` — 数据质量校验
- [ ] CPT 语料 3 种混合比例，每种 1B+ tokens
- [ ] SFT 数据 2000-5000 条，分 A/B/C 三档
- [ ] 偏好数据 2000+ 对，长度偏差可控

---

## 验收标准

- [ ] 清洗 Pipeline 代码可复用（换领域只改配置）
- [ ] 每个数据集有统计报告（token 分布、质量分布、去重率）
- [ ] `validate_data.py` 全部通过
