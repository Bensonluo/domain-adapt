"""
Phase 1 Week 15: 偏好数据切分 (train / holdout)

从 week14 的 train.jsonl (1399 对) 里确定性切出 ~100 对 holdout，供偏好胜率评估。
主训练用 train_split.jsonl，holdout 不进训练 → 胜率评估无数据泄漏。

幂等：train_split.jsonl / holdout.jsonl 已存在则跳过。
用法：
    python phase1/week15/prep_data.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "phase1" / "data" / "processed" / "preference" / "train.jsonl"
OUT_DIR = SRC.parent
TRAIN_SPLIT = OUT_DIR / "train_split.jsonl"
HOLDOUT = OUT_DIR / "holdout.jsonl"

HOLDOUT_SIZE = 100
SEED = 123


def main() -> None:
    if TRAIN_SPLIT.exists() and HOLDOUT.exists():
        print(f"[prep] 已存在，跳过: {TRAIN_SPLIT.name}, {HOLDOUT.name}")
        return

    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[prep] 源数据 {len(rows)} 对 <- {SRC}")

    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    holdout_idx = set(idx[:HOLDOUT_SIZE])

    holdout = [rows[i] for i in sorted(holdout_idx)]
    train = [rows[i] for i in range(len(rows)) if i not in holdout_idx]

    TRAIN_SPLIT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    HOLDOUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in holdout) + "\n", encoding="utf-8"
    )
    print(f"[prep] ✓ train_split={len(train)} -> {TRAIN_SPLIT}")
    print(f"[prep] ✓ holdout   ={len(holdout)} -> {HOLDOUT}")


if __name__ == "__main__":
    main()
