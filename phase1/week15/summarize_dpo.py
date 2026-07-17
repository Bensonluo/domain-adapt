"""
Phase 1 Week 15: DPO sweep 汇总

聚合 3 个 β 的: 末步 reward margin/accuracy (loss_log) + domain gain/forgetting (eval)
              + 偏好胜率/长度分桶 (winrate) → 一张表, 选最优 β。

选优逻辑 (对齐 week15 目标「对齐强度 vs 通用遗忘 trade-off」):
  1. 过滤 medical_cn Δ ≥ -0.02 (DPO 没把医疗搞崩; 阈值宽容, week12 LoRA-CPT 噪声 ~±0.04)
  2. 在 survivors 里按 matched-bucket mean-logp win-rate 排序 (长度控制后的真对齐)
  3. general_cn 遗忘作 tiebreak
  → 输出最优 β 作为 week16 (IPO/失败模式) / week17 (GRPO) 的对齐起点。

用法: python phase1/week15/summarize_dpo.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "phase1" / "results" / "week15_dpo"
BETAS = ["0.1", "0.3", "0.5"]
MED_CATASTROPHE = -0.02   # medical_cn Δ 低于此 = DPO 搞崩医疗, 排除


def last_log_row(beta_dir: Path) -> dict:
    """读 loss_log.csv 末步 (含 reward margin/accuracy)。"""
    f = beta_dir / "loss_log.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("rewards/margins") not in (None, "")]
    return rows[-1] if rows else {}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    domain = load_json(SWEEP / "domain_gain.json") or {}
    forget = load_json(SWEEP / "forgetting.json") or {}
    winrate = load_json(SWEEP / "winrate.json") or {}

    rows = []
    for b in BETAS:
        beta_dir = SWEEP / f"beta_{b}"
        lr = last_log_row(beta_dir)
        dg = (domain.get("betas") or {}).get(b)
        fg = (forget.get("betas") or {}).get(b)
        wr = winrate.get(f"dpo_b{b}", {})
        buckets = wr.get("buckets", {})
        matched = buckets.get("matched", {})
        rows.append({
            "beta": b,
            "margin": _f(lr.get("rewards/margins")),
            "acc": _f(lr.get("rewards/accuracies")),
            "med_delta": dg,
            "gen_delta": fg,
            "sum_wr": wr.get("sum_logp_winrate"),
            "mean_wr": wr.get("mean_logp_winrate"),
            "matched_sum_wr": matched.get("sum_wr"),
            "matched_mean_wr": matched.get("mean_wr"),
        })

    # 打印表
    hdr = f"{'β':>4} {'margin':>8} {'acc':>5} {'medΔ':>7} {'genΔ':>7} {'sumWR':>6} {'meanWR':>7} {'m_sumWR':>8} {'m_meanWR':>9}"
    print("\n" + "=" * 78)
    print("DPO β-sweep 汇总 (base = week12_lora_cpt/50_50_fused)")
    print("=" * 78)
    print(hdr)
    print("-" * 78)
    # base 行 (winrate 有 base)
    bw = winrate.get("base_50_50", {})
    if bw:
        print(f"{'base':>4} {'-':>8} {'-':>5} {'-':>7} {'-':>7} "
              f"{_s(bw.get('sum_logp_winrate')):>6} {_s(bw.get('mean_logp_winrate')):>7} "
              f"{_s(bw.get('buckets',{}).get('matched',{}).get('sum_wr')):>8} "
              f"{_s(bw.get('buckets',{}).get('matched',{}).get('mean_wr')):>9}   (未对齐基线)")
    for r in rows:
        print(f"{r['beta']:>4} {_s(r['margin']):>8} {_s(r['acc']):>5} {_s(r['med_delta']):>7} "
              f"{_s(r['gen_delta']):>7} {_s(r['sum_wr']):>6} {_s(r['mean_wr']):>7} "
              f"{_s(r['matched_sum_wr']):>8} {_s(r['matched_mean_wr']):>9}")

    # 选优
    survivors = [r for r in rows if r["med_delta"] is None or r["med_delta"] >= MED_CATASTROPHE]
    # 长度控制后的真对齐 = matched bucket mean-logp WR
    survivors.sort(key=lambda r: (r["matched_mean_wr"] if r["matched_mean_wr"] is not None else -1), reverse=True)
    best = survivors[0] if survivors else None

    print("\n" + "-" * 78)
    if best:
        print(f"★ 最优 β = {best['beta']} (medical_cn Δ={_s(best['med_delta'])} 未崩, "
              f"matched-bucket mean-WR={_s(best['matched_mean_wr'])} 最高 → 真对齐非长度黑客)")
        print(f"  → week16 (IPO/失败模式) / week17 (GRPO) 的对齐起点: beta_{best['beta']}_fused")
    else:
        print("✗ 无 β 通过 medical_cn 不崩门槛 — 全部排查 (可能 lr/β 需调, 或方案 B 先 SFT)")

    out = {"rows": rows, "best_beta": best["beta"] if best else None,
           "base_winrate": winrate.get("base_50_50")}
    (SWEEP / "sweep_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] → {SWEEP / 'sweep_summary.json'}")


def _f(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def _s(x):
    return "-" if x is None else x


if __name__ == "__main__":
    main()
