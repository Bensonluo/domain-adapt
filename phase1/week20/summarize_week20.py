"""
Phase1 Week20: 蒸馏深度专题结果汇总 (Part A Logit-KD + Part B On-Policy)

读各臂 eval 产物 → week20_summary.json + 对照表 (含 base + week19 response + GRPO).

读入 (run_dpo_eval / eval_cmexam / train run_config / prepare 产物):
  {sweep}/domain_gain.json, forgetting.json        — CMMLU medical/general Δ
  {sweep}/{variant}_fused/cmexam_holdout.json      — CMExam holdout acc
  {sweep}/base_cmexam_holdout.json                 — base CMExam (0.512)
  {sweep}/{variant}/run_config.json                — KD α/T 或 SFT 配置
  {sweep}/data/{variant}_sft.jsonl                 — on-policy fallback 率 (source 字段)

输出:
  {sweep}/week20_summary.json  — arms[] schema 镜像 week19 distill_summary.json (逐字段可比)
  stdout — 大对照表 (Part A KD / Part B on-policy / week19 response / GRPO)

Usage:
  python phase1/week20/summarize_week20.py --sweep phase1/results/week20_distill \
      --runs kd_t2 kd_t5 kd_pure rs_mcq rs_teacher rs_both
"""

import argparse
import json
from pathlib import Path

# base 基线 (week12 50_50_fused, week15/17/19 同口径)
BASE_CMEXAM = 0.512
BASE_CMMED = 0.5663
BASE_CMGEN = 0.6675

# 对照行 (同 base 同 eval 口径, 直接比)
GRPO_REF = {"label": "GRPO(w17)", "cmexam": 0.534, "d_cm": 0.022, "med": 0.5687, "d_med": 0.0024, "gen": 0.6725, "d_gen": 0.0050}
W19 = {  # week19 response 蒸馏 (hard label), 从 distill_summary.json 读, 兜底硬编码
    "real":    {"label": "real(w19)",    "cmexam": 0.536, "d_cm": 0.024, "med": 0.5413, "d_med": -0.0250, "gen": 0.6625, "d_gen": -0.0050},
    "distill": {"label": "distill(w19)", "cmexam": 0.530, "d_cm": 0.018, "med": 0.5650, "d_med": -0.0013, "gen": 0.6750, "d_gen": 0.0075},
}


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def kd_hyperparams(sweep, v):
    """从 run_config.json 提取 KD α/T (KD 臂) 或标记 on-policy。"""
    rc = read_json(sweep / v / "run_config.json")
    if not rc:
        return {}
    if rc.get("method", "").startswith("logit_distillation"):
        return {"alpha": rc.get("alpha"), "temperature": rc.get("temperature"),
                "kd_method": "logit_KD"}
    return {"kd_method": "onpolicy_SFT", "data_source": rc.get("data", "").split("/")[-1]}


def fallback_rate(sweep, v):
    """on-policy 臂: teacher_fallback 占比 (source 字段统计)。"""
    p = sweep / "data" / f"{v}_sft.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return None
    fb = sum(1 for r in rows if r.get("source") == "teacher_fallback")
    return round(fb / len(rows), 4)


def main():
    ap = argparse.ArgumentParser(description="Week20 蒸馏深度专题汇总")
    ap.add_argument("--sweep", default="phase1/results/week20_distill")
    ap.add_argument("--runs", nargs="*", default=["kd_t2", "kd_t5", "kd_pure", "rs_mcq", "rs_teacher", "rs_both"])
    args = ap.parse_args()
    sweep = Path(args.sweep)

    dg = read_json(sweep / "domain_gain.json") or {}
    fg = read_json(sweep / "forgetting.json") or {}
    med_deltas = dg.get("runs", {})
    gen_deltas = fg.get("runs", {})
    base_med = dg.get("base_medical_cn", BASE_CMMED)
    base_gen = fg.get("base_general_cn", BASE_CMGEN)
    base_cmexam = (read_json(sweep / "base_cmexam_holdout.json") or {}).get("accuracy", BASE_CMEXAM)

    # week19 实测值覆盖兜底硬编码 (若 summary 存在)
    w19s = read_json("phase1/results/week19_distill/distill_summary.json")
    if w19s:
        for a in w19s.get("arms", []):
            if a["variant"] in W19:
                W19[a["variant"]].update(
                    cmexam=a.get("cmexam_holdout"), d_cm=a.get("cmexam_delta"),
                    med=a.get("cmmlu_medical"), d_med=a.get("cmmlu_medical_delta"),
                    gen=a.get("cmmlu_general"), d_gen=a.get("cmmlu_general_delta"))

    arms = []
    for v in args.runs:
        cm = read_json(sweep / f"{v}_fused/cmexam_holdout.json") or {}
        cm_acc = cm.get("accuracy")
        dmed = med_deltas.get(v)
        dgen = gen_deltas.get(v)
        hp = kd_hyperparams(sweep, v)
        arm = {
            "variant": v,
            "cmexam_holdout": cm_acc,
            "cmexam_delta": round(cm_acc - base_cmexam, 4) if cm_acc is not None else None,
            "cmmlu_medical": round(base_med + dmed, 4) if dmed is not None else None,
            "cmmlu_medical_delta": dmed,
            "cmmlu_general": round(base_gen + dgen, 4) if dgen is not None else None,
            "cmmlu_general_delta": dgen,
            "cmexam_n": cm.get("n"),
        }
        arm.update(hp)
        if v.startswith("rs_"):
            arm["teacher_fallback_rate"] = fallback_rate(sweep, v)
        arms.append(arm)

    summary = {
        "base": "phase1/results/week12_lora_cpt/50_50_fused",
        "method": "week20_distillation_deep_dive (PartA logit_KD + PartB on_policy)",
        "teacher": "Qwen3-30B-A3B-Instruct-2507-MLX-4bit (mlx_lm, 4bit)",
        "n_train": 2000,
        "base_metrics": {"cmexam_holdout": base_cmexam, "cmmlu_medical": base_med, "cmmlu_general": base_gen},
        "teacher_train_acc_vs_gold": 0.892,   # extract_teacher_logits top1==gold (天花板参考)
        "references": {"grpo": GRPO_REF, "week19_response": W19},
        "arms": arms,
    }
    (sweep / "week20_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 大对照表 ──
    def row(label, cm, dcm, med, dmed, gen, dgen, note=""):
        f = lambda x, d=None, sign=True: ("  -   " if x is None else f"{x:.4f}")
        fd = lambda d: ("  -   " if d is None else f"{d:+.4f}")
        print(f"{label:<16} {f(cm):>8} {fd(dcm):>8} {f(med):>9} {fd(dmed):>8} {f(gen):>9} {fd(dgen):>8}  {note}")

    print("\n" + "=" * 96)
    print(f"  Week20 蒸馏深度专题 (base = 50_50_fused | teacher 30B MLX | n_train=2000)")
    print("=" * 96)
    print(f"{'臂':<16} {'CMExam':>8} {'Δ':>8} {'CMMLU医':>9} {'Δ':>8} {'CMMLU通':>9} {'Δ':>8}")
    print("-" * 96)
    row("base", base_cmexam, None, base_med, None, base_gen, None)
    # Part A (KD)
    for a in arms:
        if a["variant"].startswith("kd_"):
            note = f"α={a.get('alpha')} T={a.get('temperature')}"
            row(a["variant"], a["cmexam_holdout"], a["cmexam_delta"], a["cmmlu_medical"],
                a["cmmlu_medical_delta"], a["cmmlu_general"], a["cmmlu_general_delta"], note)
    # Part B (on-policy)
    for a in arms:
        if a["variant"].startswith("rs_"):
            fb = a.get("teacher_fallback_rate")
            note = f"fb={fb}" if fb is not None else ""
            row(a["variant"], a["cmexam_holdout"], a["cmexam_delta"], a["cmmlu_medical"],
                a["cmmlu_medical_delta"], a["cmmlu_general"], a["cmmlu_general_delta"], note)
    print("-" * 96)
    # 对照行
    for key in ("real", "distill"):
        r = W19[key]
        row(r["label"], r["cmexam"], r["d_cm"], r["med"], r["d_med"], r["gen"], r["d_gen"], "(week19 hard label)")
    row(GRPO_REF["label"], GRPO_REF["cmexam"], GRPO_REF["d_cm"], GRPO_REF["med"],
        GRPO_REF["d_med"], GRPO_REF["gen"], GRPO_REF["d_gen"], "(RL 对照)")
    print("=" * 96)
    print(f"  teacher train acc vs gold = 0.892 (Part A 天花板参考; holdout teacher acc ≈0.865 week19)")
    print(f"\n[✓] week20_summary.json → {sweep}/week20_summary.json")


if __name__ == "__main__":
    main()
