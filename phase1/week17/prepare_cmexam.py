"""
Phase 1 Week 17: CMExam → GRPO 格式转换

输入 (agent 已下, fzkuji/CMExam 镜像, hf-mirror):
  phase1/data/processed/cmexam/{train,test}.jsonl
  字段: Question(str) / Options(list[{key,value}]) / Answer(str) / Explanation(str)
  纯简体; Answer 多数单字母 "A"-"E", 少量多选 "ACDE"。

输出:
  grpo_train.jsonl — GRPO 训练 prompt (默认 8K, 从 train.jsonl 单选题子采样)
  holdout.jsonl    — CMExam holdout 答对率 eval (默认 500, 从 test.jsonl, 全程未训)
  格式: {"prompt": "题干\\nA. ...\\nB. ...\\n...\\n答案：", "answer": "D"}

设计:
  - 只留单选 (len(Answer)==1): 多选 MCQ-accuracy reward 不适用 (逐条单字母比对)
  - prompt 末尾 "答案：" 引导模型首字符出字母 (与 reward_functions._FIRST 正则对齐)
  - train/holdout 分流到不同原始 split (holdout 用 test, GRPO 只训 train) → 防泄漏
  - 8K 子集先验机制 (plan: 60K 全量待机制确认后 stretch)

Usage:
    python phase1/week17/prepare_cmexam.py                    # 默认 8K train + 500 holdout
    python phase1/week17/prepare_cmexam.py --train-size 16000 # 更大子集
"""

import argparse
import json
import random
from pathlib import Path

CMEXAM_DIR = Path("phase1/data/processed/cmexam")


def fmt_prompt(q: dict) -> str:
    """题干 + 候选项 + "答案：" 引导。"""
    opts = "\n".join(f"{o['key']}. {o['value']}" for o in q["Options"])
    return f"{q['Question']}\n{opts}\n答案："


def load_single_answer(path: Path) -> list[dict]:
    """读 jsonl, 只留单选题 (len(Answer)==1) + 字段齐全。"""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("Question") or not r.get("Options") or not r.get("Answer"):
            continue
        if len(r["Answer"].strip()) != 1:
            continue                       # 多选剔除 (reward 不适用)
        ans = r["Answer"].strip().upper()
        if ans not in "ABCDE":             # 防御: 个别非 A-E 字母
            continue
        out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(description="CMExam → GRPO 格式")
    p.add_argument("--train-size", type=int, default=8000, help="GRPO 训练 prompt 数 (从 train 单选)")
    p.add_argument("--holdout-size", type=int, default=500, help="holdout 答对率 eval 数 (从 test 单选)")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    train_src = load_single_answer(CMEXAM_DIR / "train.jsonl")
    hold_src = load_single_answer(CMEXAM_DIR / "test.jsonl")
    rng = random.Random(args.seed)
    rng.shuffle(train_src)
    rng.shuffle(hold_src)

    train_n = min(args.train_size, len(train_src))
    hold_n = min(args.holdout_size, len(hold_src))

    def write(path: Path, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                rec = {"prompt": fmt_prompt(r), "answer": r["Answer"].strip().upper()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_train = CMEXAM_DIR / "grpo_train.jsonl"
    out_hold = CMEXAM_DIR / "holdout.jsonl"
    write(out_train, train_src[:train_n])
    write(out_hold, hold_src[:hold_n])

    print(f"[prepare] train 单选 src={len(train_src)} → GRPO train {train_n} → {out_train}")
    print(f"[prepare] holdout 单选 src(=test, 全程未训)={len(hold_src)} → holdout {hold_n} → {out_hold}")
    print(f"[prepare] sample:\n{json.dumps({'prompt': fmt_prompt(train_src[0]), 'answer': train_src[0]['Answer'].strip()}, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
