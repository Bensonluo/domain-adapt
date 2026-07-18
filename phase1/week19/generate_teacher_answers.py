"""
Phase 1 Week 19: Teacher 答案批量生成 (mlx_lm 直跑, 不依赖 LM Studio)

读 prepare_distill_data.py 产出的 teacher_questions.jsonl, 对每题用本地 MLX teacher
(Qwen3-30B-A3B-Instruct-2507-MLX-4bit) 生成答案。

★ 不走 LM Studio HTTP (实测 502 server-wide 故障)。直接 mlx_lm.load 一次, 逐题 generate。
  enable_thinking=False → Qwen3 出字母在前+简短解释 (与 real 臂密度匹配)。

特性:
  - **resume**: 读已有输出, 跳过已答 question_id (4-7h 长跑必备)
  - **增量落盘**: 每答一题 append+flush, crash-safe
  - **<think> 兜底**: strip_think 剥除残留思考块 (thinking-off 后通常无, 兜底)
  - **字母合规 + 准确率统计**: extract teacher_letter, 报 letter_matches_gold (蒸馏天花板)
  - **smoke**: --limit 50 先测吞吐 → 决定 N
  - **MTL_TIMEOUT=0**: 防 Apple Silicon Metal 超时 (week11 CPT 同款坑)

输出 teacher_answers.jsonl:
  {question_id, gold, teacher_answer, teacher_letter, letter_matches_gold,
   elapsed_s, n_out_tokens}

Usage:
    # smoke (50 题, 测吞吐+准确率)
    python phase1/week19/generate_teacher_answers.py \\
        --teacher ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit \\
        --questions phase1/results/week19_distill/data/teacher_questions.jsonl \\
        --out phase1/results/week19_distill/data/teacher_answers.jsonl --limit 50
    # full (resume)
    python phase1/week19/generate_teacher_answers.py ... --resume
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Apple Silicon Metal: 禁超时 (week11 CPT 同款坑, 长跑 inference 必备)
os.environ.setdefault("MTL_TIMEOUT", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "week17"))
from reward_functions import extract_answer  # noqa: E402  (与训练/eval 同解析器)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
SYS_MSG = "你是医学考试助手。回答选择题时，必须先单独给出答案字母（A/B/C/D/E），再用1-2句话简要解释。"


def strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "<think>" in text:                      # 未闭合 (截断)
        text = text.split("<think>", 1)[0]
    return text.strip()


def load_done(path: Path) -> set[int]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])
    return done


def main():
    p = argparse.ArgumentParser(description="Teacher 答案批量生成 (mlx_lm 直跑)")
    p.add_argument("--teacher", required=True, help="MLX teacher 模型目录")
    p.add_argument("--questions", required=True, help="teacher_questions.jsonl")
    p.add_argument("--out", required=True, help="teacher_answers.jsonl (append+resume)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0=greedy (默认, 确定性蒸馏); >0 采样")
    p.add_argument("--limit", type=int, default=0, help="smoke: 只跑前 N 题 (0=全部)")
    p.add_argument("--resume", action="store_true", help="跳过 out 里已有的 question_id")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    import mlx_lm
    from mlx_lm.sample_utils import make_sampler   # 0.31.3: temperature 经 sampler 传, 不再是裸 kwarg

    t_load = time.time()
    model, tokenizer = mlx_lm.load(args.teacher)
    print(f"[teacher] 加载 {args.teacher} ({time.time()-t_load:.1f}s)", flush=True)

    # greedy (temp=0): 确定性、可复现、取 teacher 众数答案 = 蒸馏标准选择
    sampler = make_sampler(temp=args.temperature)
    print(f"[teacher] sampler: temp={args.temperature} (0=greedy)", flush=True)

    qs = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    done = load_done(Path(args.out)) if args.resume else set()
    todo = [q for q in qs if q["question_id"] not in done]
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"[teacher] 总题 {len(qs)} | 已答 {len(done)} (resume) | 本次 todo {len(todo)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_correct = 0
    n_compliant = 0
    tot_out_tok = 0
    tot_elapsed = 0.0
    t_start = time.time()
    with open(args.out, "a", encoding="utf-8") as fout:
        for q in todo:
            qid = q["question_id"]
            gold = q["gold"]
            msgs = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": q["teacher_user_msg"]}]
            prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                   enable_thinking=False, tokenize=False)
            t0 = time.time()
            try:
                raw = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                                      sampler=sampler, verbose=False)
            except Exception as e:
                print(f"[teacher] ✗ qid={qid} generate 异常, 跳过: {e}", flush=True)
                continue
            dt = time.time() - t0
            clean = strip_think(raw)
            letter = extract_answer(clean)
            matches = (letter == gold) if letter else False
            compliant = bool(re.match(r"^\s*[（(]?\s*[A-Ea-e]\b", clean))
            n_out = len(tokenizer.encode(clean))
            rec = {"question_id": qid, "gold": gold, "teacher_answer": clean,
                   "teacher_letter": letter, "letter_matches_gold": matches,
                   "elapsed_s": round(dt, 2), "n_out_tokens": n_out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            tot_out_tok += n_out
            tot_elapsed += dt
            if matches:
                n_correct += 1
            if compliant:
                n_compliant += 1
            if n_done % args.log_every == 0 or n_done == len(todo):
                wall = time.time() - t_start
                tps = tot_out_tok / tot_elapsed if tot_elapsed > 0 else 0
                acc = n_correct / n_done
                print(f"[teacher] {n_done}/{len(todo)} | tok/s={tps:.1f} | "
                      f"compliant={n_compliant/n_done:.2f} | teacher_acc={acc:.3f} | "
                      f"wall={wall:.0f}s", flush=True)

    wall = time.time() - t_start
    tps = tot_out_tok / tot_elapsed if tot_elapsed > 0 else 0
    print(f"\n[teacher] ✓ 本次生成 {n_done} 条 → {args.out}", flush=True)
    if n_done:
        print(f"[teacher] tok/s={tps:.1f} | compliant={n_compliant/n_done:.3f} | "
              f"★ teacher_acc(对gold)={n_correct/n_done:.3f} (= distill 臂天花板) | wall={wall:.0f}s", flush=True)
        avg_dt = tot_elapsed / n_done
        for N in (500, 1000, 2000):
            print(f"[teacher]   外推 N={N} ≈ {N*avg_dt/3600:.1f}h (单题均值 {avg_dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
