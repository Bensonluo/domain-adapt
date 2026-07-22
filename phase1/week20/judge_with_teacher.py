"""
Phase1 Week20 Part B: teacher MLX judge (RLAIF) — 给 student sample 打 1-5 分

student on-policy 采样 (generate_student_samples.py 产物) → teacher 30B 逐条评分
(医学正确性 + 解释质量)。用于 rs_teacher / rs_both 臂的 best-of-N 选择信号。

★ 设计要点:
  - **盲评 (不给 gold)**: teacher 仅看题 + student 答案, 凭自身知识评分。
    若给 gold → score 退化为 letter==gold (与 rs_mcq 重复, 无软信号)。
    盲评才能产生 rule 抓不到的信号 (解释质量 + teacher 认同度)。
  - **score-first**: 先输出 "分数：X" 再理由 → 解析稳定 (regex 抓首位 1-5)。
  - **rubric 锚定 1-5** (5=正确且详尽 ... 1=错误且无分析): 防「全 3 分」偏置。
  - report score 分布: 若 smoke 显示严重偏置 (某分 >60%) → 加 few-shot 锚定。

环境: 4bit-QLoRA-post-training venv (mlx_lm 0.31.3, 与 week19 teacher 同). --resume 断点续跑.

Usage:
  VENV=/Users/luopeng/Documents/GitHub/4bit-QLoRA-post-training/venv/bin/python
  # SMOKE (50 samples)
  $VENV phase1/week20/judge_with_teacher.py \\
    --teacher ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit \\
    --samples phase1/results/week20_distill/data/student_samples.jsonl \\
    --out phase1/results/week20_distill/data/judge_scores.jsonl --limit 50
  # FULL (--resume)
  $VENV phase1/week20/judge_with_teacher.py ... --resume
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MTL_TIMEOUT", "0")  # Apple Silicon Metal 超时禁 (week19 同款)

SYS_MSG = (
    "你是资深医学考官。评估学生答案时, 必须先单独输出评分行 '分数：X'"
    "(X 为 1-5 的整数), 再换行用一句话给出理由。严格按此格式。"
)

RUBRIC = """评分标准：
5 = 答案正确, 且解释准确、专业、详尽
4 = 答案正确, 解释合理但不够深入
3 = 答案正确但无解释, 或答案错误但分析有一定道理
2 = 答案错误, 解释有明显缺陷
1 = 答案错误且无合理分析

题目：
{prompt}

学生答案：
{text}

请评分。"""

# 分数解析: 优先 "分数：X"; 退化到首行首位 1-5; 再退化到文中首个 1-5
_SCORE_KW = re.compile(r"分数[：:]\s*([1-5])")
_LEAD_SCORE = re.compile(r"^\s*([1-5])\b")
_ANY_SCORE = re.compile(r"([1-5])")


def parse_score(text: str) -> int | None:
    if not text:
        return None
    m = _SCORE_KW.search(text)
    if m:
        return int(m.group(1))
    for line in text.splitlines():
        m = _LEAD_SCORE.search(line)
        if m:
            return int(m.group(1))
    m = _ANY_SCORE.search(text)
    return int(m.group(1)) if m else None


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["question_id"], r["sample_idx"]))
    return done


def main():
    p = argparse.ArgumentParser(description="teacher MLX judge 1-5 (RLAIF)")
    p.add_argument("--teacher", required=True, help="MLX teacher 模型目录")
    p.add_argument("--samples", required=True, help="student_samples.jsonl")
    p.add_argument("--out", required=True, help="judge_scores.jsonl (append+resume)")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0, help="0=greedy (评分确定性)")
    p.add_argument("--limit", type=int, default=0, help="smoke: 只跑前 N 个 sample (0=全部)")
    p.add_argument("--resume", action="store_true", help="跳过 (qid, sample_idx) 已有")
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    t_load = time.time()
    model, tokenizer = mlx_lm.load(args.teacher)
    print(f"[judge] 加载 teacher {args.teacher} ({time.time()-t_load:.1f}s)", flush=True)
    sampler = make_sampler(temp=args.temperature)

    # flatten student_samples → (qid, sample_idx, prompt, text)
    items = []
    for line in Path(args.samples).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for idx, s in enumerate(r["samples"]):
            items.append((r["question_id"], idx, r["prompt"], s["text"]))
    print(f"[judge] 总 samples={len(items)}", flush=True)

    done = load_done(Path(args.out)) if args.resume else set()
    todo = [it for it in items if (it[0], it[1]) not in done]
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"[judge] 已完成 {len(done)} (resume) | todo {len(todo)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_parsed = 0
    score_hist = Counter()
    tot_elapsed = 0.0
    t_start = time.time()

    with open(args.out, "a", encoding="utf-8") as fout:
        for qid, sidx, prompt, text in todo:
            user = RUBRIC.format(prompt=prompt, text=text)
            msgs = [{"role": "system", "content": SYS_MSG},
                    {"role": "user", "content": user}]
            mprompt = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)
            t0 = time.time()
            try:
                raw = mlx_lm.generate(model, tokenizer, prompt=mprompt,
                                      max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            except Exception as e:
                print(f"[judge] ✗ qid={qid} idx={sidx} generate 异常, 跳过: {e}", flush=True)
                continue
            dt = time.time() - t0
            score = parse_score(raw)
            reason = raw.strip()
            if score is not None:
                n_parsed += 1
                score_hist[score] += 1
            fout.write(json.dumps({
                "question_id": qid, "sample_idx": sidx,
                "score": score, "reason": reason, "elapsed_s": round(dt, 2),
            }, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            tot_elapsed += dt

            if n_done % args.log_every == 0 or n_done == len(todo):
                wall = time.time() - t_start
                dist = ", ".join(f"{k}:{score_hist[k]}" for k in sorted(score_hist))
                print(f"[judge] {n_done}/{len(todo)} | parsed={n_parsed/n_done:.2f} | "
                      f"score_mean={sum(k*v for k,v in score_hist.items())/max(n_parsed,1):.2f} | "
                      f"[{dist}] | {tot_elapsed/n_done:.1f}s/条 | wall={wall:.0f}s", flush=True)

    wall = time.time() - t_start
    print(f"\n[judge] ✓ 本次 {n_done} 条 → {args.out}", flush=True)
    if n_done:
        mean = sum(k * v for k, v in score_hist.items()) / max(n_parsed, 1)
        dist = ", ".join(f"{k}:{score_hist[k]}({score_hist[k]/n_parsed:.1%})" for k in sorted(score_hist))
        print(f"[judge] parsed={n_parsed}/{n_done} ({n_parsed/n_done:.3f}) | score_mean={mean:.2f} | "
              f"分布 [{dist}] | wall={wall:.0f}s", flush=True)
        # 偏置检查: 单一分数占比 > 60% 警告
        if n_parsed:
            max_frac = max(score_hist.values()) / n_parsed
            if max_frac > 0.6:
                print(f"[judge] ⚠ 单一分数占比 {max_frac:.0%} > 60% → 评分偏置, "
                      "建议加 few-shot 锚定重跑", flush=True)


if __name__ == "__main__":
    main()
