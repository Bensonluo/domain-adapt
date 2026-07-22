"""
Phase1 Week20 Part B: On-Policy Distillation (入口别名 → run_onpolicy.sh 编排)

★ 设计 (week20 plan Part B 决策):
  on-policy = student 在**自己分布**上生成 → 外部信号 (teacher judge / rule reward)
  筛选/打分 → 再学习. week18 理论: 效果最好 (student 探索自己分布, 能发现 teacher 没示范
  的好路径, 本质 RL, 能超 teacher — R1 aha moment) 但最贵.

  Mac 约束: teacher MLX 不能在线产 logprob 给 TRL GKD/DistillationTrainer (后者用
  VLLMClient 绑 vLLM + 偏 on-policy generation). 故走 **rejection-sampling SFT
  (STaR / best-of-N) + 可选 on-policy DPO** (industry standard, 复用 MLX teacher judge
  + week17 reward + week15/16 SFT/DPO 栈, 不依赖 vLLM).

  ★ judge = gpt-4o 已弃 (隐私冲突 + 国内不可达) → 本地 Qwen3-30B-A3B MLX (week19 验证).

真实实现 (本模块为编排入口, 真代码在这些文件):
  - generate_student_samples.py — student N=8 sampling (temp=0.8, resume)
  - judge_with_teacher.py       — teacher MLX judge 1-5 盲评 (RLAIF, resume)
  - prepare_onpolicy_data.py    — best-of-N → rs_mcq/rs_teacher/rs_both SFT + dpo 数据
  - run_onpolicy.sh             — 全流程 detached 编排 (generate→judge→prepare→train→eval)

三 SFT 臂 (选择信号对比):
  rs_mcq    = rule correctness (letter==gold, = GRPO 同信号)
  rs_teacher= teacher judge score (盲评软信号)
  rs_both   = correct ∩ teacher 精排 (rule 兜底 + teacher 精排)
  dpo_onpolicy (stretch) = best vs worst 偏好对 → week15 DPO 栈

用法 (本入口直接拉编排脚本, 旧文件名兼容):
  bash phase1/week20/run_onpolicy.sh
  nohup bash phase1/week20/run_onpolicy.sh > phase1/results/week20_distill/run_onpolicy.log 2>&1 &
"""

import subprocess
import sys
from pathlib import Path


def main():
    sh = Path(__file__).resolve().parent / "run_onpolicy.sh"
    print(f"[distill_on_policy] 入口别名 → 调用编排脚本: {sh}", flush=True)
    print("    真实流程: generate_student_samples → judge_with_teacher → "
          "prepare_onpolicy_data → train (week19 SFT) → fuse+eval → summarize", flush=True)
    result = subprocess.run(["bash", str(sh)])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
