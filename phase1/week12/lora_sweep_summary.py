"""
Phase 1 Week 12: LoRA-CPT sweep 汇总 — domain gain + 遗忘率 + 对照表 + 选最优
================================================================
读三比例 scores (run_lora_sweep_eval.py 产物) + base_all + 全量 FT 70-30 参考,
算每比例 medical_cn 平均 gain + general_cn 平均遗忘率, 输出对照表 + sweep_summary.json,
挑出 domain gain>0 且遗忘更轻的最优比例, 记其 fused 路径作 week15 DPO 基线。

公式 (与 eval_cpt.py / eval_forgetting.py 一致, 不是另造):
  absolute_gain = lora - base              (正 = 领域提升)
  forgetting_rate = (base - lora) / base   (正 = 遗忘)

产物 (per-ratio 格式对齐 domain_gain_real.json / forgetting_real.json):
  phase1/results/week12_lora_cpt/domain_gain_lora_{tag}.json
  phase1/results/week12_lora_cpt/forgetting_lora_{tag}.json
  phase1/results/week12_lora_cpt/sweep_summary.json

用法: .venv/bin/python phase1/week12/lora_sweep_summary.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _eval_core import TASK_GROUPS, ROOT, load_scores  # noqa: E402

SWEEP_DIR = ROOT / "phase1" / "results" / "week12_lora_cpt"
BASE_ALL = ROOT / "phase1" / "results" / "week12_lora_cpt" / "base_qwen3_17b"
# 注: 下面全量 FT 参考是旧 Qwen3.5 VLM 上的结果 (换模型前的历史锚), 与本轮 Qwen3-1.7B LoRA
# 不是同模型对照, 只作"从负 gain 走向正 gain"的进度参照, 不是严格 fair comparison。
REAL_GAIN = ROOT / "phase1" / "results" / "week12_eval" / "domain_gain_real.json"
REAL_FORGET = ROOT / "phase1" / "results" / "week12_eval" / "forgetting_real.json"
RATIOS = ["100-0", "70-30", "50-50"]


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def compute_gain(base, cpt, tasks):
    """每任务 absolute_gain + 组平均 (公式同 eval_cpt.compute_domain_gain)。"""
    gains = {}
    for t in tasks:
        if t in base and t in cpt:
            g = cpt[t] - base[t]
            gains[t] = {
                "baseline": base[t], "after_cpt": cpt[t],
                "absolute_gain": g, "direction": "improved" if g > 0 else "regressed",
            }
    return gains, avg([g["absolute_gain"] for g in gains.values()])


def compute_forgetting(base, cpt, tasks):
    """每任务 forgetting_rate + 组平均 (公式同 eval_forgetting.compute_forgetting_rate)。"""
    fr = {}
    for t in tasks:
        if t in base and t in cpt:
            r = (base[t] - cpt[t]) / base[t] if base[t] > 0 else 0
            fr[t] = {
                "before": base[t], "after": cpt[t],
                "forgetting_rate": r, "direction": "forgotten" if r > 0 else "retained",
            }
    return fr, avg([f["forgetting_rate"] for f in fr.values()])


def main():
    base = load_scores(BASE_ALL)
    med, gen = TASK_GROUPS["medical_cn"], TASK_GROUPS["general_cn"]
    ref_gain = json.loads(REAL_GAIN.read_text(encoding="utf-8"))["avg_gain"]
    ref_forget = json.loads(REAL_FORGET.read_text(encoding="utf-8"))["avg_rate"]

    rows, summary = [], {"ratios": {}, "reference_full_ft_70_30": {
        "avg_gain": ref_gain, "avg_forgetting_rate": ref_forget,
    }}

    for ratio in RATIOS:
        tag = ratio.replace("-", "_")
        eval_dir = SWEEP_DIR / f"{tag}_eval"
        if not eval_dir.exists():
            print(f"[skip] {ratio}: 无评估结果 {eval_dir} (先跑 run_lora_sweep_eval.py)")
            continue
        cpt = load_scores(eval_dir)

        gains, avg_gain = compute_gain(base, cpt, med)
        fr, avg_fr = compute_forgetting(base, cpt, gen)

        # 落盘 per-ratio (格式对齐 week12 既有 domain_gain_real / forgetting_real)
        (SWEEP_DIR / f"domain_gain_lora_{tag}.json").write_text(json.dumps({
            "task_group": ["medical_cn"], "limit": 100, "gains": gains, "avg_gain": avg_gain,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (SWEEP_DIR / f"forgetting_lora_{tag}.json").write_text(json.dumps({
            "task_group": ["general_cn"], "limit": 100, "forgetting": fr, "avg_rate": avg_fr,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        fused = str(SWEEP_DIR / f"{tag}_fused")
        rows.append((ratio, avg_gain, avg_fr, fused))
        summary["ratios"][ratio] = {
            "avg_gain": avg_gain, "avg_forgetting_rate": avg_fr, "fused_path": fused,
        }

    if not rows:
        print("❌ 没有任何比例的评估结果。先跑 run_lora_sweep_eval.py。")
        sys.exit(1)

    # 选最优: medical_cn gain 最高优先 (主目标 = 领域涨); 同 gain 遗忘更轻优先
    best = sorted(rows, key=lambda r: (-r[1], r[2]))[0]
    summary["best_ratio"] = best[0]
    summary["best_avg_gain"] = best[1]
    summary["best_avg_forgetting_rate"] = best[2]
    summary["week15_dpo_baseline"] = best[3]
    (SWEEP_DIR / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown 对照表
    print("\n" + "=" * 74)
    print("LoRA-CPT 比例 sweep 对照 (vs base; 全量 FT 70-30 作锚)")
    print("=" * 74)
    print(f"{'比例':<10} {'med_cn avg gain':>16} {'gen_cn avg 遗忘':>16}")
    print("-" * 74)
    for r, g, f, _ in rows:
        mark = "  ← 最优" if r == best[0] else ""
        print(f"{r:<10} {g:>+16.4f} {f * 100:>15.2f}%{mark}")
    print("-" * 74)
    print(f"{'全量FT 70-30':<10} {ref_gain:>+16.4f} {ref_forget * 100:>15.2f}%  (旧Qwen3.5, 不同模型)")
    print("=" * 74)
    flag = "✓ domain gain 已转正" if best[1] > 0 else "⚠ domain gain 仍为负 (考虑加 iters / rank 32)"
    print(f"\n最优比例: {best[0]}   ({flag})")
    print(f"week15 DPO 基线 (fused): {best[3]}")
    print(f"sweep_summary.json: {SWEEP_DIR / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
