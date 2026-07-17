"""
Phase 1 Week 12: LoRA-CPT 比例 sweep — fuse + 评估三比例
================================================================
对 {100-0, 70-30, 50-50} 三比例 LoRA 产物依次:
  1. fuse adapter → 独立模型 (run_mlx_evaluate 只吃独立模型)
  2. CMMLU 评估 (medical_cn 8 + general_cn 4 = 12 任务, limit=100)
base 复用 week12_lora_cpt/base_qwen3_17b (换模型后已重评; 同模型内不重跑)。

前置: 三比例 LoRA 已训练完, adapters 在
  phase1/results/week12_lora_cpt/{100_0,70_30,50_50}/adapters/
产物: phase1/results/week12_lora_cpt/{tag}_eval/scores_{tag}_fused.json

用法 (必须用 .venv/bin/python: fuse 子进程靠 sys.executable 继承解释器):
  .venv/bin/python phase1/week12/run_lora_sweep_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_core import (  # noqa: E402
    TASK_GROUPS, ROOT, resolve_tasks, run_mlx_evaluate, load_scores, fuse_model,
)

BASE = "Qwen/Qwen3-1.7B"                                    # 新 base (标准 qwen3 arch, 替换慢 VLM)
SWEEP_DIR = ROOT / "phase1" / "results" / "week12_lora_cpt"
BASE_ALL = ROOT / "phase1" / "results" / "week12_lora_cpt" / "base_qwen3_17b"
RATIOS = ["100-0", "70-30", "50-50"]
TASKS = resolve_tasks(["medical_cn", "general_cn"])         # 12 任务
LIMIT = 100


def main():
    base = load_scores(BASE_ALL)
    print(f"base (复用, 不重跑): {len(base)} tasks")

    done = {}
    for ratio in RATIOS:
        tag = ratio.replace("-", "_")
        adapter = SWEEP_DIR / tag / "adapters"
        fused = SWEEP_DIR / f"{tag}_fused"
        if not adapter.exists():
            print(f"\n[skip] {ratio}: adapter 不存在 {adapter} (还没训练?)")
            continue

        print(f"\n{'#' * 60}\n# ratio = {ratio}\n{'#' * 60}")
        fuse_model(BASE, adapter, fused)                    # 幂等 (config.json 存在则跳过)
        scores = run_mlx_evaluate(
            str(fused), TASKS, SWEEP_DIR / f"{tag}_eval", limit=LIMIT
        )
        done[ratio] = scores

        # 快速预览 (正式 gain/forgetting 见 lora_sweep_summary.py)
        print(f"--- {ratio} medical_cn (gain = lora - base; 正=提升) ---")
        for t in TASK_GROUPS["medical_cn"]:
            if t in base and t in scores:
                print(f"  {t:40s} {base[t]:.3f} -> {scores[t]:.3f}  {scores[t] - base[t]:+.3f}")

    print(f"\n✓ 评估完成 ({len(done)}/{len(RATIOS)} 比例)。")
    print("下一步: .venv/bin/python phase1/week12/lora_sweep_summary.py")


if __name__ == "__main__":
    main()
