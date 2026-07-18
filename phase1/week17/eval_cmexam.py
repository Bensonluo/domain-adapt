"""
Phase 1 Week 17: CMExam holdout 答对率 eval

对 fused GRPO 模型在 CMExam holdout (500, 全程未训) 上算 MCQ 答对率。
与训练同口径: RAW 格式 prompt (末尾 "答案："), greedy 解码, max_new_tokens=48。

设计:
  - 用 reward_functions.extract_answer 解析 (与训练 reward 完全一致 → 评估口径 = 训练信号)
  - greedy (do_sample=False) 确定性评估
  - 报告: accuracy / unparseable 率 (extract_answer 返回 None 的占比, 判模型是否乖乖出字母)
  - base baseline 同脚本跑 → GRPO 的 Δ 是干净的提升

Usage:
    # GRPO fused 模型
    python phase1/week17/eval_cmexam.py \\
        --model phase1/results/week17_grpo/mcq_base_fused \\
        --output phase1/results/week17_grpo/mcq_base_fused/cmexam_holdout.json

    # base 对照
    python phase1/week17/eval_cmexam.py \\
        --model phase1/results/week12_lora_cpt/50_50_fused \\
        --output phase1/results/week17_grpo/base_cmexam_holdout.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_functions import extract_answer   # noqa: E402 (与训练 reward 同口径)


def main():
    p = argparse.ArgumentParser(description="CMExam holdout 答对率 eval")
    p.add_argument("--model", required=True, help="fused 模型路径")
    p.add_argument("--data", default="phase1/data/processed/cmexam/holdout.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=48, help="与训练 max_completion_length 同")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只评前 N 题 (debug, 0=全部)")
    p.add_argument("--output", required=True, help="结果 json 路径")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[eval] model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("mps")
    model.eval()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    print(f"[eval] holdout {len(rows)} 题")

    tok.padding_side = "left"
    correct = 0
    unparseable = 0
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            prompts = [r["prompt"] for r in batch]
            golds = [r["answer"].strip().upper() for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to("mps")
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            gen = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for prompt, gold, text in zip(prompts, golds, gen):
                pred = extract_answer(text)
                ok = pred is not None and pred == gold
                if ok:
                    correct += 1
                if pred is None:
                    unparseable += 1
                preds.append({"gold": gold, "pred": pred, "gen_head": text[:60]})
            if (i // args.batch_size) % 10 == 0:
                acc = correct / (i + len(batch))
                print(f"  [{i+len(batch)}/{len(rows)}] running acc={acc:.3f} unparseable={unparseable}")

    n = len(rows)
    acc = correct / n
    result = {
        "model": args.model, "n": n, "accuracy": acc,
        "correct": correct, "unparseable": unparseable, "unparseable_rate": unparseable / n,
        "max_new_tokens": args.max_new_tokens,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 抽样预测落盘 (供人工 sanity)
    Path(args.output + ".preds.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in preds[:50]), encoding="utf-8")
    print(f"\n[eval] ✓ accuracy={acc:.3f} ({correct}/{n}) unparseable={unparseable} ({unparseable/n:.1%})")
    print(f"[eval] → {args.output}")


if __name__ == "__main__":
    main()
