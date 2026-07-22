"""
Phase1 Week20 Part B: on-policy best-of-N 选择 → 4 臂训练数据

join student_samples.jsonl (N=8/question) ⊕ judge_scores.jsonl (teacher 1-5) →
构造 3 个 SFT 数据集 + 1 个 DPO 数据集:

  rs_mcq    : 选择信号 = rule correctness (letter==gold). 选对的 sample (tie: 字母在前+短).
              全错 → 回退 teacher answer (week19), 保 N=2000 不缩.
  rs_teacher: 选择信号 = teacher judge score. 选最高分 sample (tie: 对的优先).
  rs_both   : correctness 硬筛 ∩ teacher 精排. 仅在 correct 里选最高分; 无 correct → 回退 teacher.
  dpo_onpolicy: chosen = best (rs_both 逻辑) / rejected = worst (min score, 错的优先).

★ 与 Part A / week19 同 base 同题集同 eval → on-policy vs off-policy 干净对照.
★ 选择信号对比: rule (客观, =GRPO 同信号) vs teacher (软, 主观认同) vs 混合.

输出 (phase1/results/week20_distill/data/):
  rs_mcq_sft.jsonl, rs_teacher_sft.jsonl, rs_both_sft.jsonl
    行: {prompt, completion, question_id, gold, source}  (source=student|teacher_fallback)
  dpo_onpolicy.jsonl
    行: {prompt, chosen, rejected, question_id, chosen_score, rejected_score}

Usage:
  phase1/.venv/bin/python phase1/week20/prepare_onpolicy_data.py \
    --samples .../student_samples.jsonl --scores .../judge_scores.jsonl \
    --teacher-answers phase1/results/week19_distill/data/teacher_answers.jsonl \
    --out-dir phase1/results/week20_distill/data
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

_LEAD = re.compile(r"^\s*[（(]?\s*[A-Ea-e]\b")


def lead_letter(text: str) -> bool:
    return bool(text and _LEAD.match(text))


def main():
    p = argparse.ArgumentParser(description="on-policy best-of-N → 4 臂数据")
    p.add_argument("--samples", required=True, help="student_samples.jsonl")
    p.add_argument("--scores", required=True, help="judge_scores.jsonl")
    p.add_argument("--teacher-answers", default="phase1/results/week19_distill/data/teacher_answers.jsonl")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    # load samples
    questions = {}   # qid → {prompt, gold, samples:[{idx,text,letter,correct}]}
    for line in Path(args.samples).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            questions[r["question_id"]] = r
    # load scores → {(qid, idx): score}
    scores = {}
    for line in Path(args.scores).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("score") is not None:
                scores[(r["question_id"], r["sample_idx"])] = r["score"]
    # load teacher answers (fallback)
    teacher_ans = {}
    tap = Path(args.teacher_answers)
    if tap.exists():
        for line in tap.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                teacher_ans[r["question_id"]] = r.get("teacher_answer", "")
    print(f"[prep] questions={len(questions)} | scores={len(scores)} | teacher_ans(fallback)={len(teacher_ans)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # enriched samples per question: add lead + len + score
    # ★ sample 无 idx 字段 (generate 输出只有 text/letter/correct) → enumerate 赋 idx,
    #   与 judge_with_teacher.py 的 enumerate(r["samples"]) 对齐, 否则 score 全塌到 idx0
    def enrich(r):
        out = []
        for i, s in enumerate(r["samples"]):
            out.append({
                "idx": i,
                "text": s["text"], "letter": s.get("letter"), "correct": s.get("correct", False),
                "lead": lead_letter(s["text"]), "len": len(s["text"]),
                "score": scores.get((r["question_id"], i)),
            })
        return out

    # selection keys (sort desc, pick [0])
    def mcq_key(s):
        return (s["correct"], s["lead"], -s["len"])
    def teacher_key(s):
        sc = s["score"] if s["score"] is not None else -1
        return (sc, s["correct"], s["lead"], -s["len"])

    arm_files = {a: open(out_dir / f"{a}_sft.jsonl", "w", encoding="utf-8") for a in ("rs_mcq", "rs_teacher", "rs_both")}
    dpo_f = open(out_dir / "dpo_onpolicy.jsonl", "w", encoding="utf-8")

    stats = {a: Counter() for a in ("rs_mcq", "rs_teacher", "rs_both", "dpo")}
    chosen_letters = {a: Counter() for a in ("rs_mcq", "rs_teacher", "rs_both")}

    for qid, r in sorted(questions.items()):
        gold = r.get("gold", "")
        prompt = r["prompt"]
        samp = enrich(r)

        def fallback_teacher(arm):
            ta = teacher_ans.get(qid, "")
            if not ta:
                return None
            arm_files[arm].write(json.dumps({
                "prompt": prompt, "completion": ta, "question_id": qid,
                "gold": gold, "source": "teacher_fallback",
            }, ensure_ascii=False) + "\n")
            stats[arm]["fallback"] += 1
            return ta

        # rs_mcq: rule correctness
        correct = [s for s in samp if s["correct"]]
        if correct:
            best = sorted(correct, key=mcq_key, reverse=True)[0]
            arm_files["rs_mcq"].write(json.dumps({
                "prompt": prompt, "completion": best["text"], "question_id": qid,
                "gold": gold, "source": "student",
            }, ensure_ascii=False) + "\n")
            stats["rs_mcq"]["student"] += 1
            chosen_letters["rs_mcq"][best["letter"]] += 1
        else:
            fallback_teacher("rs_mcq")

        # rs_teacher: teacher score (全体, 不限 correct)
        scored = [s for s in samp if s["score"] is not None]
        if scored:
            best = sorted(scored, key=teacher_key, reverse=True)[0]
            arm_files["rs_teacher"].write(json.dumps({
                "prompt": prompt, "completion": best["text"], "question_id": qid,
                "gold": gold, "source": "student",
            }, ensure_ascii=False) + "\n")
            stats["rs_teacher"]["student"] += 1
            chosen_letters["rs_teacher"][best["letter"]] += 1
        else:
            fallback_teacher("rs_teacher")

        # rs_both: correct ∩ teacher rank
        if correct and scored:
            cscored = [s for s in correct if s["score"] is not None]
            pool = cscored if cscored else correct  # 有分数的 correct 优先, 否则 correct 里随意
            best = sorted(pool, key=teacher_key, reverse=True)[0]
            arm_files["rs_both"].write(json.dumps({
                "prompt": prompt, "completion": best["text"], "question_id": qid,
                "gold": gold, "source": "student",
            }, ensure_ascii=False) + "\n")
            stats["rs_both"]["student"] += 1
            chosen_letters["rs_both"][best["letter"]] += 1
        else:
            fallback_teacher("rs_both")

        # dpo_onpolicy: chosen (rs_both 逻辑) vs rejected (min score, 错的优先)
        if len(samp) >= 2 and scored:
            chosen = sorted(pool if (correct and scored) else samp, key=teacher_key, reverse=True)[0] \
                if (correct and scored) else sorted(scored, key=teacher_key, reverse=True)[0]
            # rejected: 最低分, 错的优先
            wrong = [s for s in samp if not s["correct"]]
            rej_pool = [s for s in wrong if s["score"] is not None] if wrong else scored
            rejected = sorted(rej_pool, key=teacher_key)[0]
            if chosen["text"].strip() != rejected["text"].strip():
                dpo_f.write(json.dumps({
                    "prompt": prompt, "chosen": chosen["text"], "rejected": rejected["text"],
                    "question_id": qid,
                    "chosen_score": chosen["score"], "rejected_score": rejected["score"],
                }, ensure_ascii=False) + "\n")
                stats["dpo"]["pairs"] += 1
            else:
                stats["dpo"]["skip_same"] += 1
        else:
            stats["dpo"]["skip_nodata"] += 1

    for f in list(arm_files.values()) + [dpo_f]:
        f.close()

    # report
    print(f"\n[prep] ✓ 数据写入 {out_dir}/", flush=True)
    for a in ("rs_mcq", "rs_teacher", "rs_both"):
        c = stats[a]
        tot = sum(c.values())
        fb = c["fallback"]
        print(f"[prep] {a}: {tot} 条 (student {c['student']} / teacher_fallback {fb} = {fb/max(tot,1):.1%}) | "
              f"chosen letter 分布 {dict(chosen_letters[a])}", flush=True)
    print(f"[prep] dpo_onpolicy: {stats['dpo']['pairs']} 对 (skip_same {stats['dpo']['skip_same']} / "
          f"skip_nodata {stats['dpo']['skip_nodata']})", flush=True)


if __name__ == "__main__":
    main()
