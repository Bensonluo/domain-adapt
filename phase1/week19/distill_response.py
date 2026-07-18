"""
Phase 1 Week 19: Response Distillation — 入口 redirect

本周 response 蒸馏已实现并跑完三对照 (real/distill/mixed)，功能拆到三个脚本，
本文件只作入口指引 (不再保留旧的 deepseek-v3/qwen2.5-3b 占位 stub)。

真实实现：
  - teacher 批量生成 (mlx_lm 直跑 Qwen3-30B-A3B)  → generate_teacher_answers.py
  - 三源 SFT 数据准备 (real/distill/mixed)         → prepare_distill_data.py
  - SFT 训练 (TRL SFTTrainer + PEFT)               → train_distill_sft.py
  - 结果汇总 (三对照 + GRPO 对照)                   → summarize_distill.py
  - 一键编排                                        → run_distill.sh

结果：phase1/results/week19_distill/distill_summary.json
详见：phase1/week19/README.md

典型流程 (smoke-first)：
    # 1. 备料 (real 不需 teacher；同时 emit teacher 题文件)
    python phase1/week19/prepare_distill_data.py --n 2000 --seed 123 \\
        --out phase1/results/week19_distill/data
    # 2. teacher 生成 (mlx_lm direct, --resume 可中断续跑)
    python phase1/week19/generate_teacher_answers.py \\
        --teacher ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit \\
        --questions phase1/results/week19_distill/data/teacher_questions.jsonl \\
        --out phase1/results/week19_distill/data/teacher_answers.jsonl --resume
    # 3. distill + mixed 数据 (读 teacher_answers)
    python phase1/week19/prepare_distill_data.py --n 2000 --seed 123 \\
        --teacher-answers phase1/results/week19_distill/data/teacher_answers.jsonl \\
        --out phase1/results/week19_distill/data
    # 4. 三臂 train + fuse + eval + 汇总
    nohup bash phase1/week19/run_distill.sh > phase1/results/week19_distill/run.log 2>&1 &
"""

import sys
from pathlib import Path


def main():
    print(__doc__)
    print(f"\n[redirect] 真实实现见同目录：")
    for f in ["generate_teacher_answers.py", "prepare_distill_data.py",
              "train_distill_sft.py", "summarize_distill.py", "run_distill.sh"]:
        p = Path(__file__).parent / f
        print(f"  {'✓' if p.exists() else '✗'} {f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
