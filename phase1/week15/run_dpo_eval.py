"""
Phase 1 Week 15/16: DPO 模型评估 (merge LoRA + HF-route eval)

为什么 HF 路由而不是 MLXLM:
  week15 DPO 产物是 torch/PEFT-LoRA adapter (在 phase1/.venv torch venv 训的),
  不是 MLX 格式。week12 的 run_mlx_evaluate 走 MLXLM (mlx venv), 吃不了 torch adapter。
  故对 base + 所有 DPO run 全部走同一条 HF 路由 (HFLM + MPS), 保证 delta 有效 (同 backend)。

复用 _eval_core.py 的 model-agnostic 部分:
  TASK_GROUPS / resolve_tasks / _install_cmmlu_local_patch / _scores_from_results / _find_acc
  只把 run_mlx_evaluate 换成 run_hf_evaluate (HFLM 替 MLXLM), fuse_model 换成 PEFT merge。

参数化 (--sweep/--base/--runs, week15/16 共用, 默认值保 week15 行为):
  --betas : week15 接口, run 名 = beta_{b}, JSON key = 裸 β, 落盘字段 "betas" (summarize_dpo 依赖)
  --runs  : week16 接口, 任意 run 名 (noise_0.3 / beta_0.01 / ipo_0.3), JSON key = run 名, 字段 "runs"
  二选一; 都不给则用 --betas 默认。给了 --runs 优先。

流程 (每个 run):
  1. merge: base + adapter → merge_and_unload → 独立 HF 模型 {run}_fused (无损)
  2. eval: run_hf_evaluate({run}_fused, medical_cn+general_cn, limit=100, 0-shot, mps)
再对 base 同样 HF 路由评一次 (delta 基准), 算 domain_gain(medical_cn) + forgetting(general_cn)。

用法:
  # week15 (默认, 向后兼容)
  python phase1/week15/run_dpo_eval.py
  python phase1/week15/run_dpo_eval.py --betas 0.3
  # week16 失败模式
  python phase1/week15/run_dpo_eval.py --sweep phase1/results/week16_failmode \
      --runs noise_0.1 noise_0.3 noise_0.5 beta_0.01 beta_10 ipo_0.3
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# 国内 HF phone-home hang, 全程离线 (同 train_dpo.py)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phase1" / "week12"))
from _eval_core import (  # noqa: E402
    TASK_GROUPS, resolve_tasks, _install_cmmlu_local_patch, _scores_from_results,
)

DEFAULT_SWEEP = ROOT / "phase1" / "results" / "week15_dpo"
DEFAULT_BASE = ROOT / "phase1" / "results" / "week12_lora_cpt" / "50_50_fused"


# ─────────────────────────────────────────────
# merge: base + adapter → 独立 HF 模型 (PEFT 无损线性合并)
# ─────────────────────────────────────────────
def merge_lora(base_dir: Path, run_dir: Path, fused_dir: Path, dtype: str = "bfloat16") -> Path:
    """base + DPO adapter → merge_and_unload → fused_dir。

    显式加载 base 再 PeftModel.from_pretrained 挂 adapter, 不依赖 adapter_config 里
    base_model_name_or_path 的相对路径解析 (CWD 无关, 更稳)。
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fused_dir = Path(fused_dir)
    if (fused_dir / "config.json").exists():
        print(f"[merge] 已存在, 跳过: {fused_dir}")
        return fused_dir
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    print(f"\n[merge] {base_dir} + {run_dir} → {fused_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch_dtype)
    model = PeftModel.from_pretrained(base_model, run_dir)
    model = model.merge_and_unload()
    model.save_pretrained(fused_dir)
    AutoTokenizer.from_pretrained(run_dir).save_pretrained(fused_dir)
    print(f"[merge] ✓ {fused_dir}")
    return fused_dir


# ─────────────────────────────────────────────
# HF-route eval (HFLM + lm_eval.simple_evaluate, 复用 cmmlu 本地 patch)
# ─────────────────────────────────────────────
def run_hf_evaluate(
    model: str,
    tasks: list[str],
    output_dir: Path,
    limit: int = 100,
    num_shots: int = 0,
    batch_size: int = 8,
    seed: int = 123,
    device: str = "mps",
    dtype: str = "auto",
) -> dict[str, float]:
    """调 lm_eval.simple_evaluate + HFLM(MPS), 返回 {task: acc}。镜像 run_mlx_evaluate。"""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _install_cmmlu_local_patch()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("\n" + "=" * 60)
    print(f"评估: {model}")
    print(f"tasks: {tasks} | limit={limit} | num_shots={num_shots} | device={device} | dtype={dtype}")
    print("=" * 60)

    lm = HFLM(pretrained=str(model), device=device, dtype=dtype, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_shots,
        limit=limit,
        apply_chat_template=False,            # raw completion (匹配 CPT/DPO 纯文本形式)
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    scores = _scores_from_results(results.get("results", {}))
    tag = Path(str(model)).name.replace("/", "_") or "model"
    (output_dir / f"scores_{tag}.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scores


def group_mean(scores: dict[str, float], group: list[str]) -> float | None:
    """组内任务平均 acc (缺失任务跳过)。"""
    vals = [scores[t] for t in group if t in scores]
    return round(statistics.mean(vals), 4) if vals else None


def main():
    ap = argparse.ArgumentParser(description="DPO 模型 merge + HF-route eval (week15/16 共用)")
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP), help="sweep 根目录 (默认 week15_dpo)")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help="base 模型路径 (默认 50_50_fused)")
    ap.add_argument("--betas", nargs="*", default=["0.1", "0.3", "0.5"],
                    help="week15 接口: run=beta_{b}, JSON key=裸 β, 落盘字段 betas")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="week16 接口: 任意 run 名 (run dir=sweep/{run}), JSON key=run 名; 给了优先于 --betas")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--num-shots", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--skip-base", action="store_true", help="跳过 base 重评 (已有 HF-route base scores 时)")
    args = ap.parse_args()

    sweep = Path(args.sweep)
    base = Path(args.base)
    base_tag = base.name                       # "50_50_fused"
    tasks = resolve_tasks(["medical_cn", "general_cn"])
    base_eval_dir = sweep / "base_hf"
    base_scores_path = base_eval_dir / f"scores_{base_tag}.json"

    # 构建 run 列表: (json_key, run_dir, fused_dir)。--runs 优先, 否则 --betas (week15 兼容)
    if args.runs:
        items = [(r, sweep / r, sweep / f"{r}_fused") for r in args.runs]
        key_field = "runs"
    else:
        items = [(b, sweep / f"beta_{b}", sweep / f"beta_{b}_fused") for b in args.betas]
        key_field = "betas"

    # 1. base 重评 (HF 路由, 与 DPO 同 backend → delta 有效)
    if not args.skip_base:
        if not base_scores_path.exists():
            run_hf_evaluate(str(base), tasks, base_eval_dir, args.limit, args.num_shots,
                            args.batch_size, dtype=args.dtype)
        else:
            print(f"[base] 已存在, 跳过: {base_scores_path}")
    base_scores = json.loads(base_scores_path.read_text(encoding="utf-8"))
    base_med = group_mean(base_scores, TASK_GROUPS["medical_cn"])
    base_gen = group_mean(base_scores, TASK_GROUPS["general_cn"])
    print(f"\n[base] medical_cn={base_med} general_cn={base_gen}")

    # 2. 每个 run: merge → eval → delta
    domain_gain, forgetting = {}, {}
    for key, run_dir, fused_dir in items:
        if not (run_dir / "adapter_model.safetensors").exists():
            print(f"\n[!] 跳过 {key}: adapter 未训完 ({run_dir}/adapter_model.safetensors 不存在)")
            continue
        merge_lora(base, run_dir, fused_dir)
        scores = run_hf_evaluate(str(fused_dir), tasks, run_dir,
                                 args.limit, args.num_shots, args.batch_size, dtype=args.dtype)
        med = group_mean(scores, TASK_GROUPS["medical_cn"])
        gen = group_mean(scores, TASK_GROUPS["general_cn"])
        domain_gain[key] = round(med - base_med, 4) if med is not None and base_med is not None else None
        forgetting[key] = round(gen - base_gen, 4) if gen is not None and base_gen is not None else None
        print(f"[{key}] medical_cn={med} (Δ{domain_gain[key]})  "
              f"general_cn={gen} (Δ{forgetting[key]})")

    # 3. 落盘 gain / forgetting (字段名随接口: week15="betas", week16="runs")
    (sweep / "domain_gain.json").write_text(
        json.dumps({"base_medical_cn": base_med, key_field: domain_gain}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (sweep / "forgetting.json").write_text(
        json.dumps({"base_general_cn": base_gen, key_field: forgetting}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[✓] domain_gain.json + forgetting.json → {sweep} (字段: {key_field})")
    print("    medical_cn Δ:", domain_gain)
    print("    general_cn Δ:", forgetting)


if __name__ == "__main__":
    main()
