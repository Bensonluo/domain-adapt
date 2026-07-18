"""
Phase 1 Week 15/16: 偏好胜率 + 长度分桶 (对齐信号 + 长度黑客检测)

在 holdout 100 对 (不进训练) 上, 对 base + 各 DPO run 算:
  Σ logp(chosen) vs Σ logp(rejected), win-rate = P(chosen 的 Σlogp > rejected 的 Σlogp)。
  纯 logprob 前向 (不生成), 无需 LLM judge。

两个胜率口径 (攻 week14 QC 发现的 93.5% 长度偏差):
  1. sum-logp win-rate: Σ logp 比较 (与 sigmoid DPO 目标一致, 但 chosen 更长天然 Σ 更负 → 偏向 rejected)
  2. mean-logp win-rate: 每 token 平均 logp (长度归一, 消除长度黑客)

长度分桶: 按 |len(chosen)-len(rejected)|/max 分 matched/mid/skewed 三档。
  若 DPO 胜率提升全来自 skewed 档 → 长度黑客; matched 档也提升 → 真对齐。
  week16 IPO 头号看点: sum-WR 是否脱离 ≈0、skewed 档 mean-WR 是否脱离 0.056。

参数化 (--sweep/--base/--runs, week15/16 共用, 默认值保 week15 行为):
  --betas : week15 接口, model key = dpo_b{b}, 路径 = sweep/beta_{b}_fused (summarize_dpo 依赖)
  --runs  : week16 接口, model key = run 名, 路径 = sweep/{run}_fused
  二选一; 都不给则用 --betas 默认。给了 --runs 优先。

依赖: run_dpo_eval.py 先跑 (产出 *_fused 独立模型)。base = 50_50_fused (本身独立)。

用法:
  # week15 (默认)
  python phase1/week15/eval_winrate.py
  python phase1/week15/eval_winrate.py --betas 0.3
  # week16 失败模式
  python phase1/week15/eval_winrate.py --sweep phase1/results/week16_failmode \
      --runs noise_0.1 noise_0.3 noise_0.5 beta_0.01 beta_10 ipo_0.3
"""

import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP = ROOT / "phase1" / "results" / "week15_dpo"
DEFAULT_BASE = ROOT / "phase1" / "results" / "week12_lora_cpt" / "50_50_fused"
HOLDOUT = ROOT / "phase1" / "data" / "processed" / "preference" / "holdout.jsonl"

# 长度差分桶 (按 |Δlen|/max)
def bucket(frac: float) -> str:
    if frac < 0.2:
        return "matched"   # 长度接近 → 胜率不受长度干扰
    if frac < 0.5:
        return "mid"
    return "skewed"        # 长度差大 → 长度黑客风险区


def completion_logp(model, tok, prompt: str, completion: str, device, max_length=2048):
    """返回 (Σ logp over completion tokens, completion token 数)。

    prompt/completion 分别 tokenize 再拼 id → 干净 BPE 边界 (无跨边界 merge)。
    bf16 logits 升 float 再 log_softmax (数值稳)。
    """
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + c_ids)[:max_length]   # 超长截断 (保 prompt + completion 开头)
    comp_start = len(p_ids)
    if comp_start >= len(input_ids):
        return 0.0, 0
    import torch
    ids = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits[0].float()            # [T, V]
    log_probs = torch.log_softmax(logits[:-1], dim=-1)   # 预测 token t 用 logits[t-1]
    targets = ids[0, 1:]
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [T-1]
    start = comp_start - 1
    n_comp = len(input_ids) - comp_start
    comp_lp = token_lp[start:start + n_comp]
    return comp_lp.sum().item(), n_comp


def load_model(path: Path, device: str, dtype: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype).to(device).eval()
    return model, tok


def eval_one(model_path: Path, pairs: list, device: str) -> dict:
    import torch
    model, tok = load_model(model_path, device)
    # 每对: chosen_sum/rej_sum, chosen_mean/rej_mean, len_diff_frac, bucket
    rows = []
    for r in pairs:
        cs, cn = completion_logp(model, tok, r["prompt"], r["chosen"], device)
        rs, rn = completion_logp(model, tok, r["prompt"], r["rejected"], device)
        if cn == 0 or rn == 0:
            continue
        c_mean, r_mean = cs / cn, rs / rn
        max_len = max(cn, rn)
        frac = abs(cn - rn) / max_len if max_len else 0.0
        rows.append({
            "sum_win": cs > rs, "mean_win": c_mean > r_mean,
            "bucket": bucket(frac), "frac": frac,
        })
    del model
    torch.mps.empty_cache() if device == "mps" else None

    def wr(key, bk=None):
        sel = [x for x in rows if bk is None or x["bucket"] == bk]
        return round(statistics.mean([1.0 if x[key] else 0.0 for x in sel]), 3) if sel else None

    return {
        "n": len(rows),
        "sum_logp_winrate": wr("sum_win"),
        "mean_logp_winrate": wr("mean_win"),
        "buckets": {
            "matched": {"n": sum(1 for x in rows if x["bucket"] == "matched"), "sum_wr": wr("sum_win", "matched"), "mean_wr": wr("mean_win", "matched")},
            "mid":     {"n": sum(1 for x in rows if x["bucket"] == "mid"),     "sum_wr": wr("sum_win", "mid"),     "mean_wr": wr("mean_win", "mid")},
            "skewed":  {"n": sum(1 for x in rows if x["bucket"] == "skewed"),  "sum_wr": wr("sum_win", "skewed"),  "mean_wr": wr("mean_win", "skewed")},
        },
    }


def main():
    ap = argparse.ArgumentParser(description="holdout 胜率 + 长度分桶 (week15/16 共用)")
    ap.add_argument("--sweep", default=str(DEFAULT_SWEEP), help="sweep 根目录 (默认 week15_dpo)")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help="base 模型路径 (默认 50_50_fused)")
    ap.add_argument("--betas", nargs="*", default=["0.1", "0.3", "0.5"],
                    help="week15 接口: model key=dpo_b{b}, 路径=sweep/beta_{b}_fused")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="week16 接口: model key=run 名, 路径=sweep/{run}_fused; 给了优先于 --betas")
    ap.add_argument("--holdout", default=str(HOLDOUT))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    sweep = Path(args.sweep)
    base = Path(args.base)
    device = args.device
    pairs = [json.loads(l) for l in Path(args.holdout).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[winrate] holdout {len(pairs)} 对 | device={device} | sweep={sweep}")

    # 构建 model dict: base + runs。base key = base_{base.name} (默认 base_50_50)
    models = {f"base_{base.name}": base}
    if args.runs:
        for r in args.runs:
            fused = sweep / f"{r}_fused"
            if (fused / "config.json").exists():
                models[r] = fused
            else:
                print(f"[winrate] 跳过 {r}: fused 模型不存在 ({fused}), 先跑 run_dpo_eval.py")
    else:
        for b in args.betas:
            fused = sweep / f"beta_{b}_fused"
            if (fused / "config.json").exists():
                models[f"dpo_b{b}"] = fused
            else:
                print(f"[winrate] 跳过 β={b}: fused 模型不存在 ({fused}), 先跑 run_dpo_eval.py")

    results = {}
    for name, path in models.items():
        print(f"\n==== {name}: {path} ====")
        results[name] = eval_one(path, pairs, device)
        r = results[name]
        print(f"  sum-logp WR={r['sum_logp_winrate']}  mean-logp WR={r['mean_logp_winrate']}  (n={r['n']})")
        for bk, bv in r["buckets"].items():
            print(f"    {bk:8s} n={bv['n']:>3}  sum_wr={bv['sum_wr']}  mean_wr={bv['mean_wr']}")

    out = sweep / "winrate.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] → {out}")


if __name__ == "__main__":
    main()
