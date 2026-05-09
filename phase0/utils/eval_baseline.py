"""
基线评估: lm-evaluation-harness 在 Qwen2.5-3B-Instruct 上
=====================================================

目标:
    - 跑通 lm-eval 环境
    - 拿到基座模型在医疗相关 MMLU 子集上的分数
    - 记录为后续所有实验的对比基准

跑法 (Mac,用 MPS 或 CPU,3B 推理慢但不需要训练):
    source .venv/bin/activate
    python phase0/utils/eval_baseline.py

输出:
    phase0/results/baseline_qwen25_3b.json

MMLU 子集说明:
    mmlu_clinical_knowledge  : 临床知识
    mmlu_anatomy             : 解剖学
    mmlu_medical_genetics    : 医学遗传学
"""

from __future__ import annotations

import json
from pathlib import Path

import lm_eval
import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "phase0" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 核心评估逻辑
# ---------------------------------------------------------------------------
def run_baseline(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    tasks: list[str] | None = None,
    batch_size: int = 4,
    output_path: Path | None = None,
) -> dict:
    """
    使用 lm-eval Python API (而非 CLI) 跑评估,方便后续在代码里复用。

    参数:
        model_name: HuggingFace model ID
        tasks     : 任务列表,默认医疗相关 MMLU 子集
        batch_size: 推理 batch size (Mac MPS 显存小,建议调小)
    """
    if tasks is None:
        tasks = [
            "mmlu_clinical_knowledge",
            "mmlu_anatomy",
            "mmlu_medical_genetics",
        ]
    if output_path is None:
        output_path = RESULTS_DIR / "baseline_qwen25_3b.json"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"评估设备: {device}")
    print(f"模型: {model_name}")
    print(f"任务: {', '.join(tasks)}")

    # lm_eval API 入口
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name},device={device}",
        tasks=tasks,
        batch_size=batch_size,
        verbosity="ERROR",
    )

    # 提取关键指标
    summary = {
        "model": model_name,
        "device": device,
        "tasks": {},
    }
    for task in tasks:
        task_res = results["results"].get(task, {})
        summary["tasks"][task] = {
            "acc": task_res.get("acc", None),
            "acc_norm": task_res.get("acc_norm", None),
        }
    # 算平均
    accs = [v["acc"] for v in summary["tasks"].values() if v["acc"] is not None]
    summary["average_acc"] = sum(accs) / len(accs) if accs else None

    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n结果已保存: {output_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    print("=" * 60)
    print("Phase 0 基线评估")
    print("=" * 60)
    run_baseline()


if __name__ == "__main__":
    main()
