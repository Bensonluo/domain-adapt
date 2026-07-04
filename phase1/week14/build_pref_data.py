"""
Phase 1 Week 14: 偏好数据集构建 + QC（替换 prep/build_preference_data.py 的 stub 角色）
=====================================================================================

数据源: 魔搭 `modelzhang/medical_evidence_DPO`（中文医疗循证 DPO，专家生成，
{prompt, chosen, rejected} 三元组）。in-domain 匹配 CPT 医疗语料，免 teacher API、
免下载大模型（stub 默认的 deepseek-v3 vs qwen2.5-3b 生成作为备选，见报告）。

本脚本只做: 加载 → 校验/过滤 → 去重 → 落盘 → QC 报告 + 100 对抽检。
不改动 prep/ 下的原 stub（保留作对照）。

用法:
    python phase1/week14/build_pref_data.py
产物:
    phase1/data/processed/preference/train.jsonl      (干净 {prompt,chosen,rejected})
    phase1/week14/pref_qc_report.json                 (QC 统计)
    phase1/week14/sampled_100.jsonl                   (人工抽检样本)
"""

import hashlib
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "phase1" / "data" / "raw" / "preference_src" / "medical_evidence_DPO" / "dpo_answer.jsonl"
OUT_DIR = ROOT / "phase1" / "data" / "processed" / "preference"
OUT_TRAIN = OUT_DIR / "train.jsonl"
QC_REPORT = ROOT / "phase1" / "week14" / "pref_qc_report.json"
SAMPLED = ROOT / "phase1" / "week14" / "sampled_100.jsonl"

BASE_MODEL = str(ROOT / "models" / "Qwen3.5-0.8B-Base-ms")
DATASET_ID = "modelzhang/medical_evidence_DPO"
SOURCE_FILE = "dpo_answer.jsonl"  # 数据集 loading script 指定的 canonical TRAIN

# 过滤阈值（字符级；DPO 文本质量底线）
MIN_LEN = 10
MAX_LEN = 20000
SAMPLE_N = 100
SEED = 123
BIAS_THRESHOLD = 0.5  # |chosen-rejected|/max > 0.5 标记为"潜在长度偏差"


# ─────────────────────────────────────────────
# 加载 + 过滤 + 去重
# ─────────────────────────────────────────────
def load_raw(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def filter_pairs(pairs: list[dict]) -> tuple[list[dict], dict]:
    """返回 (干净对, 丢弃统计)。"""
    kept, dropped = [], {}
    for p in pairs:
        keys = set(p.keys())
        if not {"prompt", "chosen", "rejected"}.issubset(keys):
            dropped["missing_fields"] = dropped.get("missing_fields", 0) + 1
            continue
        prompt, chosen, rejected = str(p["prompt"]), str(p["chosen"]), str(p["rejected"])
        if not (prompt.strip() and chosen.strip() and rejected.strip()):
            dropped["empty_field"] = dropped.get("empty_field", 0) + 1
            continue
        if chosen.strip() == rejected.strip():
            dropped["chosen_eq_rejected"] = dropped.get("chosen_eq_rejected", 0) + 1
            continue
        if any(len(x) < MIN_LEN for x in (prompt, chosen, rejected)):
            dropped["too_short"] = dropped.get("too_short", 0) + 1
            continue
        if any(len(x) > MAX_LEN for x in (prompt, chosen, rejected)):
            dropped["too_long"] = dropped.get("too_long", 0) + 1
            continue
        kept.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return kept, dropped


def dedup_exact(pairs: list[dict]) -> tuple[list[dict], int]:
    """按 (prompt, chosen, rejected) 完整三元组去重（保序，保留首条）。

    注意: **不**按 prompt 去重 —— 同一 prompt 带不同 chosen/rejected 是合法的偏好对
    (更多 preference signal), DPO 训练完全可接受。只丢完全相同的整条记录。
    """
    seen, out = set(), []
    for p in pairs:
        h = hashlib.md5((p["prompt"] + "\x1f" + p["chosen"] + "\x1f" + p["rejected"]).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return out, len(pairs) - len(out)


# ─────────────────────────────────────────────
# QC 统计
# ─────────────────────────────────────────────
def _summary(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 2),
        "median": round(st.median(xs), 2),
        "stdev": round(st.stdev(xs), 2) if len(xs) > 1 else 0.0,
        "min": round(min(xs), 2),
        "max": round(max(xs), 2),
        "p95": round(_percentile(xs, 95), 2),
        "p99": round(_percentile(xs, 99), 2),
    }


def _percentile(xs: list[float], q: float) -> float:
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * (q / 100)
    lo, hi = int(k), min(int(k) + 1, len(xs2) - 1)
    return xs2[lo] + (xs2[hi] - xs2[lo]) * (k - lo)


def char_stats(pairs: list[dict]) -> dict:
    lens = {k: [len(p[k]) for p in pairs] for k in ("prompt", "chosen", "rejected")}
    diffs = [len(p["chosen"]) - len(p["rejected"]) for p in pairs]
    chosen_longer = sum(1 for d in diffs if d > 0)
    biased = sum(
        1 for p in pairs
        if abs(len(p["chosen"]) - len(p["rejected"])) / max(len(p["chosen"]), len(p["rejected"]), 1) > BIAS_THRESHOLD
    )
    return {
        "char_lengths": {k: _summary(v) for k, v in lens.items()},
        "chosen_minus_rejected": _summary(diffs),
        "chosen_longer_pct": round(chosen_longer / len(pairs) * 100, 2),
        "length_bias_pct": round(biased / len(pairs) * 100, 2),  # |diff|/max > 0.5 的占比
        "bias_threshold": BIAS_THRESHOLD,
    }


def token_stats(pairs: list[dict]) -> dict:
    """用 base tokenizer 算 token 数（best-effort: 失败则返回 None）。"""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as e:  # pragma: no cover
        return {"available": False, "error": f"{type(e).__name__}: {e}"}

    def tok_len(s: str) -> int:
        return len(tok.encode(s, add_special_tokens=False))

    # DPO 实际喂入是 prompt+response 拼接，算拼接长度更有用（max_seq 规划）
    chosen_tok = [tok_len(p["prompt"]) + tok_len(p["chosen"]) for p in pairs]
    rejected_tok = [tok_len(p["prompt"]) + tok_len(p["rejected"]) for p in pairs]
    prompt_tok = [tok_len(p["prompt"]) for p in pairs]
    return {
        "available": True,
        "tokenizer": "Qwen3.5-0.8B-Base-ms",
        "prompt_tokens": _summary(prompt_tok),
        "prompt_plus_chosen_tokens": _summary(chosen_tok),
        "prompt_plus_rejected_tokens": _summary(rejected_tok),
        "max_seq_coverage": {
            "1024": round(sum(1 for x in chosen_tok if x <= 1024) / len(chosen_tok) * 100, 2),
            "2048": round(sum(1 for x in chosen_tok if x <= 2048) / len(chosen_tok) * 100, 2),
            "4096": round(sum(1 for x in chosen_tok if x <= 4096) / len(chosen_tok) * 100, 2),
        },
    }


def write_sample(pairs: list[dict], n: int, seed: int) -> list[dict]:
    random.seed(seed)
    sample = random.sample(pairs, min(n, len(pairs)))
    with open(SAMPLED, "w", encoding="utf-8") as f:
        for p in sample:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return sample


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    if not SRC.exists():
        raise SystemExit(f"❌ 源数据不存在: {SRC}\n先 git clone medical_evidence_DPO 数据集")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(SRC)
    print(f"[load] {SOURCE_FILE}: {len(raw)} 条原始")

    kept, dropped = filter_pairs(raw)
    deduped, n_dup = dedup_exact(kept)
    print(f"[filter] 丢弃 {sum(dropped.values())} 条: {dropped}")
    print(f"[dedup] 完全重复 {n_dup} 条 → 最终 {len(deduped)} 对")

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[write] {OUT_TRAIN.relative_to(ROOT)}")

    sample = write_sample(deduped, SAMPLE_N, SEED)
    print(f"[sample] {len(sample)} 对 → {SAMPLED.relative_to(ROOT)}")

    report = {
        "dataset": {
            "id": DATASET_ID,
            "source_file": SOURCE_FILE,
            "license_note": "card frontmatter 标 MIT, loading script 标 cc-by-nc-sa-4.0 (个人学习可用, 商用前需澄清)",
            "format": "{prompt, chosen, rejected}",
            "domain": "中文医疗循证 (clinical guidelines / disease mechanism / pharmacology / diagnostics / research methods)",
        },
        "counts": {
            "raw": len(raw),
            "after_filter": len(kept),
            "after_exact_dedup": len(deduped),
            "dropped_by_filter": dropped,
            "exact_duplicates_removed": n_dup,
            "note": "同 prompt 不同答案的偏好对保留 (DPO 合法); 仅去完全相同的整条记录",
            "target_was_2000": len(deduped) >= 2000,
        },
        "alternatives_in_dataset": {
            "deepseek_vs_qwen6B_DPO.jsonl": "2099 条, chat-message 格式, deepseek(teacher) vs qwen6B(student) 生成 — 即 stub 原设的 teacher/student 路线, 已预生成; 转换为 {prompt,chosen,rejected} 可用",
            "filter_dpo_dataset.jsonl": "857 条, agent/function-calling 格式 (带 tools 字段), 与本周 DPO 对齐目标不符",
        },
        "char_qc": char_stats(deduped),
        "token_qc": token_stats(deduped),
        "outputs": {
            "train": str(OUT_TRAIN.relative_to(ROOT)),
            "sampled_100": str(SAMPLED.relative_to(ROOT)),
        },
    }
    QC_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {QC_REPORT.relative_to(ROOT)}")

    # 控制台速览
    cq = report["char_qc"]
    print("\n=== QC 速览 ===")
    print(f"  最终对数: {len(deduped)} (raw {len(raw)})")
    print(f"  长度偏差: chosen 更长 {cq['chosen_longer_pct']}% | |diff|/max>{BIAS_THRESHOLD} 占 {cq['length_bias_pct']}%")
    if report["token_qc"].get("available"):
        t = report["token_qc"]["max_seq_coverage"]
        print(f"  max_seq 覆盖: 1024→{t['1024']}%  2048→{t['2048']}%  4096→{t['4096']}% (prompt+chosen)")


if __name__ == "__main__":
    main()
