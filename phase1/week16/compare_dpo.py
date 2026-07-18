"""
Phase 1 Week 16: DPO 跨设定对比 + 失败模式分析 (Day1-2 + Day3-4 交付物)

聚合 week15 sigmoid 三β + week16 失败模式 6 run → 一张总表 + 失败模式 Δ 分析。
读各 sweep 目录已产出的 JSON (domain_gain/forgetting/winrate) + 每 run 的 loss_log.csv +
sweep_run.log (训练 wall-clock), 不重跑任何训练/评估。

设计 (合并了原计划的 summarize_failmodes: 一个脚本产 comparison.md + failure_summary.json,
避免两个聚合器重复):
  - 控制基线 = week15 sigmoid β=0.3 (noise=0, loss=sigmoid), 已有产物。
  - 每个 week16 失败模式 run 求 Δ vs 控制基线 (sum/mean-WR、matched/skewed 长度桶、漂移)。
  - 三类失败模式各一段判定 (数据驱动: 看 Δ 符号 + 量级)。

JSON 字段兼容: week15 domain_gain/forgetting 用 "betas" (key=裸β), week16 用 "runs" (key=run名);
winrate week15 key=dpo_b{b}, week16 key=run名。本脚本按字段名自动选映射。

用法:
  python phase1/week16/compare_dpo.py                    # 默认读 week15_dpo + week16_failmode
  python phase1/week16/compare_dpo.py --sweeps phase1/results/week16_failmode
"""

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "phase1" / "results"
DEFAULT_SWEEPS = [RESULTS / "week15_dpo", RESULTS / "week16_failmode"]
# 控制基线 (所有失败模式的对照): week15 sigmoid β=0.3
CONTROL = ("week15_dpo", "beta_0.3")
OUT_DIR = ROOT / "phase1" / "results" / "week16_failmode"   # comparison.md / failure_summary.json 落这


# ─────────────────────────────────────────────
# 加载单个 run 的指标
# ─────────────────────────────────────────────
def _f(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def last_log_row(run_dir: Path) -> dict:
    f = run_dir / "loss_log.csv"
    if not f.exists():
        return {}
    with open(f, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("rewards/margins") not in (None, "")]
    return rows[-1] if rows else {}


def load_run_config(run_dir: Path) -> dict:
    f = run_dir / "run_config.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def discover_runs(sweep_dir: Path) -> list[str]:
    """sweep 下所有有 loss_log.csv 的子目录名 (run 名)。"""
    if not sweep_dir.exists():
        return []
    return sorted([d.name for d in sweep_dir.iterdir()
                   if d.is_dir() and (d / "loss_log.csv").exists()])


def key_for_run(run_name: str, field: str) -> str:
    """domain_gain/forgetting JSON 里的 key: week15(betas)=去 beta_ 前缀, week16(runs)=原名。"""
    if field == "betas":
        return run_name.removeprefix("beta_")
    return run_name


def wr_key_for_run(run_name: str, field: str) -> str:
    """winrate JSON 里的 key: week15=dpo_b{b}, week16=原名。"""
    if field == "betas":
        return f"dpo_b{run_name.removeprefix('beta_')}"
    return run_name


def detect_field(gain_json: dict) -> str:
    return "runs" if "runs" in gain_json else "betas"


def build_row(sweep_name: str, run_name: str, sweep_dir: Path,
              gain: dict, forget: dict, winrate: dict, field: str, train_min) -> dict:
    run_dir = sweep_dir / run_name
    cfg = load_run_config(run_dir)
    lr = last_log_row(run_dir)
    k = key_for_run(run_name, field)
    wk = wr_key_for_run(run_name, field)
    wr = winrate.get(wk, {})
    buckets = wr.get("buckets", {})
    margin = _f(lr.get("rewards/margins"))
    beta = cfg.get("beta")
    drift = round(margin / beta, 2) if (margin is not None and beta) else None
    return {
        "sweep": sweep_name, "run": run_name,
        "mode": classify(run_name, cfg),
        "beta": beta, "noise": cfg.get("noise", 0.0), "loss_type": cfg.get("loss_type", "sigmoid"),
        "med_delta": gain.get(field, {}).get(k),
        "gen_delta": forget.get(field, {}).get(k),
        "sum_wr": wr.get("sum_logp_winrate"), "mean_wr": wr.get("mean_logp_winrate"),
        "matched_mean_wr": buckets.get("matched", {}).get("mean_wr"),
        "skewed_mean_wr": buckets.get("skewed", {}).get("mean_wr"),
        "skewed_n": buckets.get("skewed", {}).get("n"),
        "margin": margin, "acc": _f(lr.get("rewards/accuracies")), "drift": drift,
        "train_min": train_min,
    }


def classify(run_name: str, cfg: dict) -> str:
    if run_name == "beta_0.3" and cfg.get("noise", 0) == 0 and cfg.get("loss_type", "sigmoid") == "sigmoid":
        return "control"
    if cfg.get("noise", 0) > 0:
        return "noise"
    if cfg.get("loss_type") == "ipo":
        return "ipo"
    if run_name.startswith("beta_") and cfg.get("beta", 0.3) in (0.01, 10, 0.1, 0.5):
        # week15 β 扫描归 baseline-scan; week16 极端 β 归 extreme-beta
        return "extreme-beta" if cfg.get("beta") in (0.01, 10) else "beta-scan"
    return "other"


# ─────────────────────────────────────────────
# 训练 wall-clock: 解析 sweep_run.log 的 [HH:MM:SS] ---- TRAIN name ---- / ✓ TRAIN name OK
# ─────────────────────────────────────────────
_LOG_TRAIN = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\] ---- TRAIN (\S+)")
_LOG_OK = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] [✓✗] TRAIN (\S+) (?:OK|FAIL)")


def _to_sec(hhmmss: str) -> int:
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_train_minutes(sweep_dir: Path) -> dict[str, float]:
    """从 sweep_run.log 抽每个 run 的训练 wall-clock (分钟)。同名 run 取首对。"""
    log = sweep_dir / "sweep_run.log"
    if not log.exists():
        return {}
    starts: dict[str, int] = {}
    out: dict[str, float] = {}
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _LOG_TRAIN.match(line)
        if m and m.group(2) not in starts:
            starts[m.group(2)] = _to_sec(m.group(1))
            continue
        m = _LOG_OK.search(line)
        if m and m.group(1) in starts:
            # 用本行时间戳作结束 (OK/FAIL 行有 [HH:MM:SS])
            ts = re.match(r"^\[(\d{2}:\d{2}:\d{2})\]", line)
            if ts:
                dur = _to_sec(ts.group(1)) - starts[m.group(1)]
                if dur < 0:
                    dur += 86400   # 跨午夜
                out[m.group(1)] = round(dur / 60.0, 1)
            starts.pop(m.group(1), None)
    return out


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def gather(sweep_paths: list[Path]) -> list[dict]:
    rows = []
    for sp in sweep_paths:
        sweep_name = sp.name
        gain = load_json(sp / "domain_gain.json")
        forget = load_json(sp / "forgetting.json")
        winrate = load_json(sp / "winrate.json")
        field = detect_field(gain) if gain else "runs"
        train_min_map = parse_train_minutes(sp)
        for run_name in discover_runs(sp):
            rows.append(build_row(sweep_name, run_name, sp, gain, forget, winrate, field,
                                  train_min_map.get(run_name)))
    return rows


def find_control(rows: list[dict]) -> dict | None:
    for r in rows:
        if r["sweep"] == CONTROL[0] and r["run"] == CONTROL[1]:
            return r
    # 兜底: 任何 mode==control
    return next((r for r in rows if r["mode"] == "control"), None)


def delta_vs_control(run: dict, ctrl: dict) -> dict:
    keys = ["sum_wr", "mean_wr", "matched_mean_wr", "skewed_mean_wr", "drift", "med_delta", "gen_delta", "acc"]
    d = {}
    for k in keys:
        a, b = run.get(k), ctrl.get(k)
        d[k] = round(a - b, 3) if (a is not None and b is not None) else None
    return d


def fmt(x, spec="") -> str:
    if x is None:
        return "—"
    return format(x, spec) if spec else str(x)


MODE_LABEL = {
    "control": "控制基线", "beta-scan": "β扫描(week15)",
    "noise": "噪声剂量", "extreme-beta": "极端β", "ipo": "IPO", "other": "其他",
}


def render_md(rows: list[dict], ctrl: dict | None) -> str:
    lines = []
    lines.append("# Week16：DPO 跨设定对比 + 失败模式分析\n")
    lines.append(f"> 控制基线 = week15 sigmoid β=0.3 (noise=0, 300对, 1 epoch)。"
                 f"所有 week16 run 从此基线分支, 每次只改一个变量。\n")
    lines.append("> margin=β×Δlogratio 跨 β 不可直比, 看漂移=margin/β。WR=holdout 100 对。\n---\n")

    # §1 总表
    lines.append("## §1 总表 (week15 β扫描 + week16 失败模式)\n")
    lines.append("| sweep | run | 模式 | β | noise | loss | medΔ | genΔ | sumWR | meanWR | m_meanWR | s_meanWR | acc | 漂移 | 训练min |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['sweep']} | {r['run']} | {MODE_LABEL.get(r['mode'], r['mode'])} "
            f"| {fmt(r['beta'])} | {fmt(r['noise'])} | {r['loss_type']} "
            f"| {fmt(r['med_delta'], '+.3f')} | {fmt(r['gen_delta'], '+.3f')} "
            f"| {fmt(r['sum_wr'])} | {fmt(r['mean_wr'])} | {fmt(r['matched_mean_wr'])} "
            f"| {fmt(r['skewed_mean_wr'])} | {fmt(r['acc'])} | {fmt(r['drift'])} | {fmt(r['train_min'])} |"
        )
    lines.append("")

    if not ctrl:
        lines.append("\n> ⚠ 未找到控制基线 (week15 beta_0.3), 跳过失败模式 Δ 分析。\n")
        return "\n".join(lines)

    # §2 失败模式 Δ vs 控制
    lines.append("## §2 失败模式 Δ vs 控制基线 (week15 sigmoid β=0.3)\n")
    lines.append(f"> 控制: sumWR={fmt(ctrl['sum_wr'])} meanWR={fmt(ctrl['mean_wr'])} "
                 f"matched_meanWR={fmt(ctrl['matched_mean_wr'])} skewed_meanWR={fmt(ctrl['skewed_mean_wr'])} "
                 f"漂移={fmt(ctrl['drift'])} acc={fmt(ctrl['acc'])}\n")
    lines.append("| run | 模式 | ΔsumWR | ΔmeanWR | Δmatched | Δskewed | Δ漂移 | Δacc |")
    lines.append("|---|---|---|---|---|---|---|---|")
    fm_rows = [r for r in rows if r["sweep"] == "week16_failmode"]
    for r in fm_rows:
        d = delta_vs_control(r, ctrl)
        lines.append(
            f"| {r['run']} | {MODE_LABEL.get(r['mode'], r['mode'])} "
            f"| {fmt(d['sum_wr'], '+.3f')} | {fmt(d['mean_wr'], '+.3f')} "
            f"| {fmt(d['matched_mean_wr'], '+.3f')} | {fmt(d['skewed_mean_wr'], '+.3f')} "
            f"| {fmt(d['drift'], '+.2f')} | {fmt(d['acc'], '+.3f')} |"
        )
    lines.append("")

    # §3 三类判定 (数据驱动提示)
    lines.append("## §3 失败模式判定 (数据驱动)\n")
    verdicts = verdict_section(fm_rows, ctrl)
    lines.extend(verdicts)
    return "\n".join(lines)


def verdict_section(fm_rows: list[dict], ctrl: dict) -> list[str]:
    out = []
    by_mode = {}
    for r in fm_rows:
        by_mode.setdefault(r["mode"], []).append(r)

    # noise dose-response
    out.append("### ① 噪声剂量 (β=0.3 sigmoid, noise∈{0.1,0.3,0.5})\n")
    noises = sorted(by_mode.get("noise", []), key=lambda r: r["noise"])
    if noises:
        out.append("| noise | acc | Δacc vs控制 | sumWR | meanWR | ΔmeanWR |")
        out.append("|---|---|---|---|---|---|")
        for r in noises:
            d = delta_vs_control(r, ctrl)
            out.append(f"| {fmt(r['noise'])} | {fmt(r['acc'])} | {fmt(d['acc'], '+.3f')} "
                       f"| {fmt(r['sum_wr'])} | {fmt(r['mean_wr'])} | {fmt(d['mean_wr'], '+.3f')} |")
        # 判定: acc 是否随噪声降 (有鲁棒性) vs 仍冲高 (强拟合噪声=过拟合)
        accs = [r["acc"] for r in noises if r["acc"] is not None]
        ctrl_acc = ctrl["acc"]
        if accs and ctrl_acc:
            drop = ctrl_acc - min(accs)
            out.append(f"\n- 判定线索: 控制 acc={fmt(ctrl_acc)}, 噪声组最低 acc={fmt(min(accs))} "
                       f"(降 {fmt(drop, '+.3f')})。降越多→对噪声敏感(非纯记忆); 不降→强拟合含矛盾标签=过拟合。\n")
    else:
        out.append("(无 noise run)\n")

    # extreme beta
    out.append("\n### ② 极端 β (sigmoid, β∈{0.01, 10})\n")
    exts = sorted(by_mode.get("extreme-beta", []), key=lambda r: r["beta"])
    if exts:
        out.append("| β | 漂移(margin/β) | acc | medΔ | meanWR |")
        out.append("|---|---|---|---|---|")
        for r in exts:
            out.append(f"| {fmt(r['beta'])} | {fmt(r['drift'])} | {fmt(r['acc'])} "
                       f"| {fmt(r['med_delta'], '+.3f')} | {fmt(r['mean_wr'])} |")
        out.append("\n- 判定线索: β=0.01→近无 KL 约束, 漂移应巨大(或梯度太小不学, acc≈0.5); "
                   "β=10→强锚定, 漂移应极小、acc 可能上不去。\n")
    else:
        out.append("(无 extreme-β run)\n")

    # IPO
    out.append("\n### ③ IPO (β=0.3, length-normalized) — 攻 week15 长度偏差\n")
    ipos = by_mode.get("ipo", [])
    if ipos:
        r = ipos[0]
        d = delta_vs_control(r, ctrl)
        out.append(f"- IPO: sumWR={fmt(r['sum_wr'])} (控制 {fmt(ctrl['sum_wr'])}, Δ{fmt(d['sum_wr'], '+.3f')}); "
                   f"skewed 档 meanWR={fmt(r['skewed_mean_wr'])} (控制 {fmt(ctrl['skewed_mean_wr'])}, "
                   f"Δ{fmt(d['skewed_mean_wr'], '+.3f')}); matched 档 meanWR={fmt(r['matched_mean_wr'])} "
                   f"(控制 {fmt(ctrl['matched_mean_wr'])}, Δ{fmt(d['matched_mean_wr'], '+.3f')})。")
        out.append("- 判定线索: IPO 长度归一 score → 若 sumWR/skewed_meanWR 相对 sigmoid **回升** "
                   "(脱离 ≈0 / 0.056) = length-norm 对症; 若不变 = τ=0.3 偏低或长度偏差非 loss 形式可解。\n")
    else:
        out.append("(无 ipo run)\n")
    return out


def main():
    ap = argparse.ArgumentParser(description="DPO 跨设定对比 + 失败模式分析")
    ap.add_argument("--sweeps", nargs="+", default=[str(p) for p in DEFAULT_SWEEPS],
                    help="sweep 目录列表 (默认 week15_dpo + week16_failmode)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="comparison.md / failure_summary.json 输出目录")
    args = ap.parse_args()

    sweep_paths = [Path(s) for s in args.sweeps]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = gather(sweep_paths)
    ctrl = find_control(rows)

    # 控制 + 总表 控制台预览
    print(f"\n[compare] 收集 {len(rows)} 个 run")
    if ctrl:
        print(f"[compare] 控制基线 = {ctrl['sweep']}/{ctrl['run']} "
              f"(sumWR={fmt(ctrl['sum_wr'])} meanWR={fmt(ctrl['mean_wr'])} 漂移={fmt(ctrl['drift'])})")
    for r in rows:
        print(f"  - {r['sweep']}/{r['run']:14s} [{r['mode']:12s}] "
              f"medΔ={fmt(r['med_delta'], '+.3f')} sumWR={fmt(r['sum_wr'])} "
              f"meanWR={fmt(r['mean_wr'])} acc={fmt(r['acc'])}")

    # 1. markdown 报告
    md = render_md(rows, ctrl)
    md_path = out_dir / "week16_dpo_comparison.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\n[✓] 报告 → {md_path}")

    # 2. failure_summary.json (机器可读)
    summary = {
        "control": {"sweep": ctrl["sweep"], "run": ctrl["run"]} if ctrl else None,
        "rows": rows,
        "failure_delta_vs_control": [
            {"run": r["run"], "mode": r["mode"], **delta_vs_control(r, ctrl)}
            for r in rows if r["sweep"] == "week16_failmode" and ctrl
        ],
    }
    (out_dir / "failure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[✓] 机器可读 → {out_dir / 'failure_summary.json'}")


if __name__ == "__main__":
    main()
