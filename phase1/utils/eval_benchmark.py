"""
Phase 1 Utils: Benchmark 评估工具

统一的 benchmark 评估脚本，支持 lm-eval-harness。

Usage:
    python phase1/utils/eval_benchmark.py --model ./results/model/ --tasks medical
    python phase1/utils/eval_benchmark.py --model ./results/model/ --tasks mmlu
"""

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_lm_eval(model_path: str, tasks: list[str], batch_size: int = 8) -> dict:
    """运行 lm-eval-harness"""
    task_str = ",".join(tasks)
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task_str,
        "--batch_size", str(batch_size),
        "--output_path", f"phase1/results/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    ]

    # TODO: 运行命令并解析结果
    print(f"Would run: {' '.join(cmd)}")
    return {}


def compare_models(models: dict[str, str], tasks: list[str]) -> dict:
    """对比多个模型的 benchmark 结果"""
    results = {}
    for name, path in models.items():
        print(f"Evaluating {name} ({path})...")
        results[name] = run_lm_eval(path, tasks)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"],
                       help="Task names: medical, mmlu, all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--compare", nargs="+", help="Additional model paths to compare")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    # Task group presets
    TASK_GROUPS = {
        "medical": ["mmlu_medical_genetics", "mmlu_anatomy", "mmlu_clinical_knowledge",
                     "mmlu_professional_medicine", "mmlu_college_medicine"],
        "mmlu": ["mmlu"],
        "all": ["mmlu", "mmlu_medical_genetics", "mmlu_anatomy", "mmlu_clinical_knowledge"],
    }

    tasks = []
    for t in args.tasks:
        tasks.extend(TASK_GROUPS.get(t, [t]))

    results = run_lm_eval(args.model, tasks, args.batch_size)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Evaluated {args.model} on {tasks}")


if __name__ == "__main__":
    main()
