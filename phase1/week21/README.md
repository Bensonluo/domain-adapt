# Week 21：合成数据生成 + 数据质量评估

> 方法学纠错见 [`CORRECTIONS.md`](CORRECTIONS.md)。当前结论保持“有希望但证据不足”：点估计满足阈值，但统计非劣效未建立。

> 目标: 实现 Self-Instruct + Evol-Instruct pipeline，评估合成数据质量。
> 预计时间: 14-20 小时

> **思考锚点**: "合成数据能替代 50% 真实数据吗？质量损失有多大？"

---

## Day 1-2: Self-Instruct Pipeline

### 做什么
1. 准备 100 条种子指令
2. 大模型生成新问题 + 回答
3. 过滤低质量生成

### 跑
```bash
python phase1/week21/self_instruct.py \
    --seeds phase1/results/week21_synthetic/data/seed_instructions.jsonl \
    --model ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit \
    --backend mlx --n 1000 --batch-size 4 --resume \
    --output phase1/results/week21_synthetic/data/
```

---

## Day 3: Evol-Instruct

### 做什么
1. 实现问题复杂度演化：简单 → 复杂
2. 多步演化（depth 1-3）

### 跑
```bash
python phase1/week21/evol_instruct.py \
    --input phase1/results/week21_synthetic/data/self_instruct.jsonl \
    --model ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit \
    --backend mlx --depth 3 --limit 250 --resume \
    --output phase1/results/week21_synthetic/data/
```

---

## Day 4-5: 质量评估 + 替代实验

### 做什么
1. 合成数据质量评估：
   - 多样性（BERTScore / Self-BLEU）
   - 正确性（固定样本复核；本次为 Codex 手工审阅，非临床医生/人类标注）
   - 与真实数据的分布差异
2. **关键实验**: 合成数据替代 50% 真实数据，看效果是否保持

### 交付物

- [x] Self-Instruct pipeline 代码
- [x] Evol-Instruct pipeline 代码
- [x] 质量评估结构化结果 — `results/week21_synthetic/week21_summary.json` 及 lineage/QC 产物
- [x] 替代实验结构化结果 — 逐题预测、配对统计与 `week21_summary.json`

## 可复现实验

本周默认离线使用本地 Qwen3 MLX teacher，不向外部 API 发送医学数据。完整流程按输入、参数和产物内容哈希安全续跑：

```bash
bash phase1/week21/run_week21.sh
```

新生成的数据会在 30 条固定审阅样本产生后暂停；逐条填写 `review_correct`、`review_notes`、`reviewer` 和 `reviewer_kind`（`ai_nonclinician`、`human_nonclinician` 或 `human_clinician`）后，再次运行同一命令即可继续。不要修改 `content_sha256`；流水线会据此防止题目内容变化后沿用旧标签。当前仓库中的审阅记录是 Codex 的手工 AI 复核，不得表述为人类或临床审核。

流水线把 raw response、accepted records、reject reasons、模型路径和 backend 分开记录；`mock` backend 只供单测，完整验收会拒绝 mock 证据。关键对照固定为：

- 历史 control：Week 19 `real`，2,000 条真实 SFT，复用 CMExam development 500；
- 历史 treatment：1,000 真实 + 1,000 合成；该历史结果保留但不再作为确认实验输入；
- 操作性点估计阈值：treatment 相对 control 降幅不超过 2 个百分点；正式非劣效结论还要求配对置信区间下界高于 −2 个百分点。

### 修复后的确认数据（尚未重训）

确认实验改用嵌套匹配的 versioned 数据，不覆盖历史产物：control 为 2,000 条真实题，treatment 保留其中 1,000 条并用 1,000 条同答案标签、近 completion-token 长度的合成题替换。两臂标签分布相同，completion token 总预算差 0.86%；对冻结确认候选做字符 3-gram Jaccard 检查，阈值 `0.78` 下两臂 prohibited overlap 都为 0。

```bash
phase1/.venv/bin/python phase1/week21/prepare_clean_matched_replacement.py \
  --real-raw phase1/data/processed/cmexam/train.jsonl \
  --synthetic phase1/results/week21_synthetic/data/evol_instruct.jsonl \
              phase1/results/week21_synthetic/data/self_instruct.jsonl \
  --confirmation phase1/data/processed/cmexam/confirmation_candidate_v1.jsonl \
  --control-output phase1/data/processed/confirmation/week21_control_real_clean_v1.jsonl \
  --treatment-output phase1/data/processed/confirmation/week21_treatment_synthetic50_clean_v1.jsonl \
  --audit-output phase1/audit/week21_clean_replacement_audit.json \
  --n-total 2000 --synthetic-fraction 0.5 --real-pool-size 4000 \
  --similarity-threshold 0.78 \
  --tokenizer phase1/results/week12_lora_cpt/50_50_fused --seed 123
```

审计记录见 [`../audit/week21_clean_replacement_audit.json`](../audit/week21_clean_replacement_audit.json)。它只证明确认数据的隔离与可比性，不把尚未执行的多 seed 重训写成已完成结果。

快速检查：

```bash
python -m unittest discover -s phase1/week21/tests -v
python phase1/week21/validate_week21.py --scope code
python phase1/week21/validate_week21.py --scope complete
```

---

## 自测题

1. **Self-Instruct 生成的指令质量受什么因素影响最大？**
2. **Evol-Instruct 的"演化"在做什么？为什么不直接生成复杂问题？**
3. **合成数据替代 50% 真实数据后，效果掉了多少？可以接受吗？**

## 实测结论（2026-08-24）

- 生成：Self-Instruct 1,000 条 + Evol-Instruct 250 条，全部来自本地 MLX teacher；accepted 数据结构有效率 100%，精确重复率 0%。
- Codex 手工 AI 复核（非临床医生/人类标注）：25/30 = 83.3%，仅作定性 sanity check。
- 分布偏移：合成题干平均长度为真实样本的 2.03 倍；答案标签 JS divergence = 0.097。
- 50% 替代：统一 CPU/float32 贪心评测下，CMExam holdout 53.0%（265/500），对比 Week 19 全真实 control 53.8%（269/500），下降 **0.8 个百分点**。
- 判定：点估计满足“下降不超过 2 个百分点”的操作性阈值；配对 bootstrap 95% 区间为 [−3.4, +1.8] 个百分点，跨越 −2 个百分点，因此统计非劣效性未建立，结论为“有希望但证据不足”（McNemar p=0.6440）。
- 泄漏检查：历史替换集有 12 条高相似真实来源题，因此历史结果不升级；修复后的 confirmation v1 两臂均已排除阈值内重叠，等待重训。
- 复现限制：本次已完成的原始生成发生在 MLX RNG 显式逐请求设种修复之前；raw response 可追溯，但不能仅凭 seed 位级复现。后续全新运行已显式设置 MLX RNG。

---

## 验收清单

- [x] Self-Instruct pipeline 完成
- [x] Evol-Instruct pipeline 完成
- [x] 合成数据质量评估完成
- [x] 替代实验完成
- [x] 有合成 vs 真实数据的量化对比
