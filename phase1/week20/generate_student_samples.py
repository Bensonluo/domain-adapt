"""
Phase1 Week20 Part B: student on-policy 采样 (N=8/question)

student = 50_50_fused (与 Part A / week19 同 base). 每题采 N=8 条 (temp=0.8),
供 on-policy 蒸馏: rejection-sampling SFT (STaR/best-of-N) + on-policy DPO.

★ on-policy 核心: student 在**自己分布**上生成 → 能探索 teacher 没示范的路径
  (week18 Part B 问题: 能否超越 teacher 天花板?).

- prompt = week19 distill_sft.jsonl 同题集 (同 Part A, 干净对照)
- letter 抽取复用 week17 reward_functions.extract_answer (byte-level 同 eval 口径)
- correct = (letter == gold)
- 输出 student_samples.jsonl: {question_id, gold, prompt, samples:[{text, letter, correct}]×8]

环境: phase1/.venv (HF transformers, MPS). 离线. --resume 断点续跑.

Usage:
  # SMOKE (10 题 × 8)
  phase1/.venv/bin/python phase1/week20/generate_student_samples.py \
    --model phase1/results/week12_lora_cpt/50_50_fused \
    --data phase1/results/week19_distill/data/distill_sft.jsonl \
    --out phase1/results/week20_distill/data/student_samples.jsonl --limit 10
  # FULL (--resume)
  ... --n 8 --resume
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MTL_TIMEOUT", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "week17"))  # reward_functions
from reward_functions import extract_answer  # noqa: E402


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description="student on-policy N=8 采样")
    p.add_argument("--model", required=True, help="50_50_fused (student)")
    p.add_argument("--data", required=True, help="week19 distill_sft.jsonl")
    p.add_argument("--out", required=True, help="student_samples.jsonl (append+resume)")
    p.add_argument("--n", type=int, default=8, help="每题采样数")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="smoke: 只跑前 N (0=全部)")
    p.add_argument("--resume", action="store_true", help="跳过 out 里已有 question_id")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    torch.manual_seed(args.seed)
    print(f"[gen] device={device} | N={args.n} | temp={args.temperature} | top_p={args.top_p} | "
          f"max_new={args.max_new_tokens}", flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成用左 pad
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"[gen] 加载 student {args.model} ({time.time()-t0:.1f}s)", flush=True)

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    done = set()
    if args.resume and Path(args.out).exists():
        for line in Path(args.out).read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])
    todo = [r for r in rows if r["question_id"] not in done]
    print(f"[gen] 总 {len(rows)} | 已完成 {len(done)} (resume) | todo {len(todo)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    n_done = 0
    n_with_correct = 0          # 至少 1 条对的题数 (rs_mcq 可行性)
    tot_samples = 0
    tot_correct = 0
    t_start = time.time()

    with open(args.out, "a", encoding="utf-8") as fout:
        for r in todo:
            qid = r["question_id"]
            gold = r.get("gold", "").upper()
            prompt = r["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.n,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            # out: (N, prompt_len + new); 去掉 prompt 部分
            plen = inputs["input_ids"].shape[1]
            samples = []
            letters = []
            for k in range(args.n):
                comp_ids = out[k, plen:].tolist()
                # 截到 eos
                if eos_id in comp_ids:
                    comp_ids = comp_ids[: comp_ids.index(eos_id)]
                text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                letter = extract_answer(text)
                correct = (letter == gold) if letter else False
                samples.append({"text": text, "letter": letter, "correct": correct})
                letters.append(letter)
                tot_samples += 1
                if correct:
                    tot_correct += 1
            has_correct = any(s["correct"] for s in samples)
            if has_correct:
                n_with_correct += 1

            fout.write(json.dumps({
                "question_id": qid, "gold": gold, "prompt": prompt,
                "samples": samples,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

            if n_done % args.log_every == 0 or n_done == len(todo):
                wall = time.time() - t_start
                acc = tot_correct / max(tot_samples, 1)
                cov = n_with_correct / n_done
                # diversity: 本批最近一题的 unique letter 数
                ul = len(set(letters))
                print(f"[gen] {n_done}/{len(todo)} | samples={tot_samples} | "
                      f"acc={acc:.3f} | 题覆盖(≥1对)={cov:.3f} | 末题diversity={ul}/{args.n} | "
                      f"wall={wall:.0f}s", flush=True)

    wall = time.time() - t_start
    print(f"\n[gen] ✓ 本次 {n_done} 题 → {args.out}", flush=True)
    if n_done:
        print(f"[gen] samples={tot_samples} (均值 {args.n}/题) | "
              f"★ overall_acc={tot_correct/tot_samples:.3f} | "
              f"★ 题覆盖(≥1对)={n_with_correct/n_done:.3f} (rs_mcq 可行性) | "
              f"wall={wall:.0f}s ({wall/max(n_done,1):.1f}s/题)", flush=True)


if __name__ == "__main__":
    main()
