"""
Phase 1 Week 19: 蒸馏三对照结果汇总

读各臂 eval 产物 → distill_summary.json + 三对照表 (含 base + week17 GRPO 对照)。

读入 (run_dpo_eval.py / eval_cmexam.py / generate_teacher_answers.py 产物):
  {sweep}/domain_gain.json   — {base_medical_cn, runs:{variant: med_delta}}
  {sweep}/forgetting.json    — {base_general_cn, runs:{variant: gen_delta}}
  {sweep}/{variant}_fused/cmexam_holdout.json — {accuracy, n, unparseable}
  {sweep}/base_cmexam_holdout.json — base CMExam acc (0.512)
  {sweep}/data/teacher_answers.jsonl — teacher 对 gold 准确率 (distill 臂天花板)

输出:
  {sweep}/distill_summary.json — 结构镜像 week17 grpo_summary.json
  stdout — 三对照表 (CMExam holdout / CMMLU medical / general, 含 Δ + GRPO 对照)

Usage:
    python phase1/week19/summarize_distill.py \\
        --sweep phase1/results/week19_distill --runs real distill mixed
"""

import argparse
import json
from pathlib import Path

VARIANTS = ["real", "distill", "mixed"]
# week17 GRPO 对照 (同 CMExam holdout 口径, 直接比 蒸馏 vs RL)
GRPO_REF = {"cmexam_holdout": 0.534, "cmexam_delta": 0.022, "label": "week17 GRPO (mcq_base_fused)"}
BASE_CMMED = 0.5663   # week12 50_50_fused 基线 (week15/17 同口径, 便于核对)
BASE_CMGEN = 0.6675
BASE_CMEXAM = 0.512


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def teacher_accuracy(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    nc = sum(1 for r in rows if r.get("letter_matches_gold"))
    comp = sum(1 for r in rows if r.get("teacher_letter"))
    return {"n": len(rows), "acc": round(nc / len(rows), 4), "compliant": round(comp / len(rows), 4)}


def main():
    ap = argparse.ArgumentParser(description="蒸馏三对照汇总")
    ap.add_argument("--sweep", default="phase1/results/week19_distill")
    ap.add_argument("--runs", nargs="*", default=VARIANTS)
    args = ap.parse_args()
    sweep = Path(args.sweep)
    runs = args.runs

    dg = read_json(sweep / "domain_gain.json") or {}
    fg = read_json(sweep / "forgetting.json") or {}
    # run_dpo_eval 用 key_field="runs" (week16/19 接口)
    med_deltas = dg.get("runs", {})
    gen_deltas = fg.get("runs", {})
    base_med = dg.get("base_medical_cn", BASE_CMMED)
    base_gen = fg.get("base_general_cn", BASE_CMGEN)

    base_cm = read_json(sweep / "base_cmexam_holdout.json") or {}
    base_cmexam = base_cm.get("accuracy", BASE_CMEXAM)

    arms = []
    for v in runs:
        cm = read_json(sweep / f"{v}_fused/cmexam_holdout.json") or {}
        cm_acc = cm.get("accuracy")
        dmed = med_deltas.get(v)
        dgen = gen_deltas.get(v)
        arms.append({
            "variant": v,
            "cmexam_holdout": cm_acc,
            "cmexam_delta": round(cm_acc - base_cmexam, 4) if cm_acc is not None else None,
            "cmmlu_medical": round(base_med + dmed, 4) if dmed is not None else None,
            "cmmlu_medical_delta": dmed,
            "cmmlu_general": round(base_gen + dgen, 4) if dgen is not None else None,
            "cmmlu_general_delta": dgen,
            "cmexam_n": cm.get("n"),
            "cmexam_unparseable": cm.get("unparseable"),
        })

    teacher = teacher_accuracy(sweep / "data/teacher_answers.jsonl")

    summary = {
        "base": "phase1/results/week12_lora_cpt/50_50_fused",
        "method": "response_distillation_SFT",
        "teacher": "Qwen3-30B-A3B-Instruct-2507-MLX-4bit (mlx_lm direct, greedy)",
        "n_train": 2000,
        "base_metrics": {"cmexam_holdout": base_cmexam, "cmmlu_medical": base_med, "cmmlu_general": base_gen},
        "teacher_accuracy_vs_gold": teacher,    # = distill 臂天花板 (insight #1)
        "grpo_reference": GRPO_REF,             # 同 holdout 口径, 蒸馏 vs RL
        "arms": arms,
    }
    (sweep / "distill_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 三对照表 (stdout) ──
    def fmt(x, base, delta_key=None):
        if x is None:
            return "  -   "
        return f"{x:.4f}"

    print("\n" + "=" * 78)
    print(f"  Week19 Response-Distillation 三对照 (base = 50_50_fused)")
    print("=" * 78)
    print(f"{'臂':<10} {'CMExam':>8} {'Δ':>8} {'CMMLU医':>9} {'Δ':>8} {'CMMLU通':>9} {'Δ':>8}")
    print("-" * 78)
    print(f"{'base':<10} {base_cmexam:>8.4f} {'—':>8} {base_med:>9.4f} {'—':>8} {base_gen:>9.4f} {'—':>8}")
    for a in arms:
        d_cm = f"{a['cmexam_delta']:+.4f}" if a['cmexam_delta'] is not None else "  -   "
        d_med = f"{a['cmmlu_medical_delta']:+.4f}" if a['cmmlu_medical_delta'] is not None else "  -   "
        d_gen = f"{a['cmmlu_general_delta']:+.4f}" if a['cmmlu_general_delta'] is not None else "  -   "
        cm = f"{a['cmexam_holdout']:.4f}" if a['cmexam_holdout'] is not None else "  -   "
        med = f"{a['cmmlu_medical']:.4f}" if a['cmmlu_medical'] is not None else "  -   "
        gen = f"{a['cmmlu_general']:.4f}" if a['cmmlu_general'] is not None else "  -   "
        print(f"{a['variant']:<10} {cm:>8} {d_cm:>8} {med:>9} {d_med:>8} {gen:>9} {d_gen:>8}")
    print("-" * 78)
    print(f"{GRPO_REF['label']:<10} {GRPO_REF['cmexam_holdout']:>8.4f} {GRPO_REF['cmexam_delta']:+8.4f}  (同 holdout 口径, 蒸馏 vs RL)")
    if teacher:
        print(f"\n  ★ teacher 对 gold 准确率 = {teacher['acc']} ({teacher['n']} 题, compliant={teacher['compliant']}) "
              f"→ distill 臂天花板")
    print(f"\n[✓] distill_summary.json → {sweep}/distill_summary.json")


if __name__ == "__main__":
    main()
