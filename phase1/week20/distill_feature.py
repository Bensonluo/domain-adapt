"""
Phase1 Week20 Part A: Feature Distillation (入口别名 → train_logit_kd.py)

★ 为什么 logit-KD 而非 hidden-state 对齐 (week20 plan Part A.1 决策):
  传统 feature distill = 中间层 hidden state 对齐 (DistilBERT, 加 projection layer).
  但**跨框架 (MLX teacher 30B + HF student 1.7B) hidden state 不可行**:
    mlx.core.Tensor vs torch.Tensor, 层命名/维度对齐复杂, 跨框架 projection 不可导.
  → 现代生成式 LLM 的 feature distill 等价 = **logit-level KD** (Hinton KD on token logits):
    学 teacher 每个 token 位置的完整 soft 概率分布 (携带类间关系的 dark knowledge),
    而非只 argmax (hard label = week19 distill 臂).

  经典 hidden-state 蒸馏 (FeatureDistillationLoss = MSE on projected hidden, 见下文注释)
  是 encoder 模型时代的做法; 对 decoder LLM + Mac 跨框架约束, logit KD 是落地等价路径.

真实实现 (本模块为入口别名, 真代码在这些文件):
  - kd_loss.py              — L = α·CE + (1-α)·T²·KL_restricted (fp32, 5 单测过)
  - extract_teacher_logits.py — teacher MLX forward 存 top-K raw logits
  - train_logit_kd.py       — Trainer + 自定义 compute_loss + PEFT-LoRA (本模块 main 别名)
  - run_feature.sh          — 三臂 (kd_t2/kd_t5/kd_pure) detached 编排

经典 hidden-state 参考 (不用于本周, 仅对照):
  class FeatureDistillationLoss:                                   # encoder 时代
      def __init__(self, t_dim, s_dim): self.proj = nn.Linear(s_dim, t_dim)
      def forward(self, t_hidden, s_hidden):
          return F.mse_loss(self.proj(s_hidden), t_hidden.detach())  # 跨框架不可导 → 弃

用法 (与 train_logit_kd.py 完全一致, 本入口保留旧文件名兼容):
  phase1/.venv/bin/python phase1/week20/distill_feature.py \\
    --model phase1/results/week12_lora_cpt/50_50_fused \\
    --data phase1/results/week19_distill/data/distill_sft.jsonl \\
    --logits phase1/results/week20_distill/data/teacher_topk_logits.jsonl \\
    --output phase1/results/week20_distill/kd_t2 --alpha 0.5 --temperature 2.0
"""

# 真实 KD loss 可从此入口直接 import (向后兼容旧 stub 的 from distill_feature import ...)
from kd_loss import kd_loss  # noqa: F401

# main 别名 → 真实训练入口 (保留文件名作为 entry point, 旧脚本无需改路径)
from train_logit_kd import main  # noqa: F401

if __name__ == "__main__":
    main()
