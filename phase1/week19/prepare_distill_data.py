"""
Phase 1 Week 19: CMExam → 蒸馏 SFT 数据准备 (受控三对照)

输入: phase1/data/processed/cmexam/train.jsonl (字段 Question/Options/Answer/Explanation)
输出 (到 --out 目录):
  real_sft.jsonl        — {prompt, completion="{Answer}\\n{Explanation}", question_id, gold}
  teacher_questions.jsonl — {question_id, gold, prompt_raw, teacher_user_msg}  (喂 generate_teacher_answers.py)
  [若给 --teacher-answers]:
  distill_sft.jsonl     — {prompt, completion="<teacher 答案, 字母在前>", question_id, gold}
  mixed_sft.jsonl       — N/2 real + N/2 distill (每题出现一次, 确定性 50/50 分配)

★ 关键设计 (Plan agent 实测): CMExam 人工 Explanation 不以字母开头 (字母藏文中 "（B对）（A错）"),
  直接当 completion 会让 eval_cmexam.extract_answer 抓错字母。故 real/distill 两臂 completion 都
  显式格式化成字母在前 → 与 week17 eval/reward 同口径。

受控三对照: 同一批 N 题 (单选 + Explanation 非空, seed 确定性), 只换 completion 来源:
  real=人写 Answer+Explanation / distill=35B-teacher 答案 / mixed=各半。
  隔离"学 teacher vs 学人"的差距, 且同 CMExam holdout 口径可直接比 week17 GRPO。

prompt 与 week17/eval **字节一致** (直接 import prepare_cmexam.fmt_prompt, 不重写)。

Usage:
    # Phase A: 备料 (real 不需 teacher; 同时 emit teacher 题文件)
    python phase1/week19/prepare_distill_data.py --n 2000 --seed 123 \\
        --out phase1/results/week19_distill/data

    # Phase B: teacher 跑完后, 生成 distill + mixed
    python phase1/week19/prepare_distill_data.py --n 2000 --seed 123 \\
        --teacher-answers phase1/results/week19_distill/data/teacher_answers.jsonl \\
        --out phase1/results/week19_distill/data
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

# 复用 week17 的 prompt 格式 + 单选过滤 (保证字节一致) 和 extract_answer (letter 解析)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "week17"))
from prepare_cmexam import fmt_prompt, load_single_answer  # noqa: E402
from reward_functions import extract_answer  # noqa: E402

CMEXAM_DIR = Path("phase1/data/processed/cmexam")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """剥除 Qwen3.6 thinking 块 (开 enable_thinking=false 后通常没有, 兜底)。"""
    text = _THINK_RE.sub("", text)
    # 未闭合的 <think> (截断): 删掉 <think> 及之后
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def real_completion(q: dict) -> str:
    """real 臂 completion: 金字母 + 换行 + 人工解释 (字母在前, eval 友好)。"""
    return f"{q['Answer'].strip().upper()}\n{q['Explanation'].strip()}"


def normalize_distill_completion(teacher_text: str, gold: str):
    """distill 臂 completion 归一化成字母在前。

    蒸馏语义: 学生学 teacher 的实际答案 (含其错误 = 蒸馏天花板=teacher 准确率)。
    - teacher 已合规 (首字符字母): 原样用。
    - 不合规: 取 teacher 解析出的字母 (extract_answer); 取不到才回退 gold (计 recovery)。
    返回 (completion, teacher_letter, status)。
    """
    text = strip_think(teacher_text).strip()
    if re.match(r"^\s*[（(]?\s*[A-Ea-e]\b", text):          # _LEAD 同款, 已字母在前
        m = re.match(r"^\s*[（(]?\s*([A-Ea-e])\b", text)
        return text, m.group(1).upper(), "compliant"
    letter = extract_answer(text)                           # teacher 文中解析的字母 (可能 None)
    status = "recovered_teacher" if letter else "recovered_gold"
    prepend = letter if letter else gold
    return f"{prepend}\n{text}", (letter if letter else gold), status


def teacher_user_msg(prompt_raw: str) -> str:
    """构造给 teacher 的指令 user message (强制字母在前、关思考、≤80字理由)。"""
    return (
        "请回答下面这道医学单项选择题。\n"
        "要求：\n"
        "1) 第一行只输出答案字母（A/B/C/D/E 中的一个），不要加标点、括号或汉字；\n"
        "2) 第二行起用 1-2 句话（不超过 80 字）简要说明理由；\n"
        "3) 不要输出思考过程、不要重复题目。\n\n"
        f"{prompt_raw}"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_questions(n: int, seed: int):
    """采样 N 道 (单选 + Explanation 非空 + 不在 test split), 确定性, 保证 0 碰撞。

    CMExam train/test split 有 ~0.02% 重复题 (实测 11/6590 test 在 train)。
    若采样进 train 的题出现在 holdout(取自 test) → 评估污染。故从 train 池里
    先剔除所有 test-split 题, 结构性保证 0 碰撞 (比事后 dedup 500 更强)。
    """
    hold_src = load_single_answer(CMEXAM_DIR / "test.jsonl")
    hold_qs = {r["Question"].strip() for r in hold_src}          # test 全量题干
    pool = [r for r in load_single_answer(CMEXAM_DIR / "train.jsonl")
            if (r.get("Explanation") or "").strip()
            and r["Question"].strip() not in hold_qs]            # 剔除 test 题
    excluded = sum(1 for r in load_single_answer(CMEXAM_DIR / "train.jsonl")
                   if (r.get("Explanation") or "").strip() and r["Question"].strip() in hold_qs)

    rng = random.Random(seed)
    rng.shuffle(pool)
    n = min(n, len(pool))
    picked = pool[:n]
    collisions = sum(1 for q in picked if q["Question"].strip() in hold_qs)
    print(f"[prepare] train 单选+Explanation非空 池 (剔除 test 重复 {excluded} 后)={len(pool)} → 采样 {n}")
    print(f"[prepare] holdout(test) 题数={len(hold_qs)}; 采样集碰撞={collisions} (结构性=0)")
    return picked


def phase_a(n: int, seed: int, out: Path) -> list[dict]:
    """emit real_sft.jsonl + teacher_questions.jsonl, 返回 picked 题列表。"""
    picked = sample_questions(n, seed)
    real_rows, tq_rows = [], []
    for i, q in enumerate(picked):
        prompt = fmt_prompt(q)
        gold = q["Answer"].strip().upper()
        real_rows.append({"prompt": prompt, "completion": real_completion(q),
                          "question_id": i, "gold": gold})
        tq_rows.append({"question_id": i, "gold": gold, "prompt_raw": prompt,
                        "teacher_user_msg": teacher_user_msg(prompt)})
    write_jsonl(out / "real_sft.jsonl", real_rows)
    write_jsonl(out / "teacher_questions.jsonl", tq_rows)
    print(f"[prepare] ✓ real_sft.jsonl ({len(real_rows)}) + teacher_questions.jsonl ({len(tq_rows)}) → {out}")
    _sanity(real_rows, label="real")
    return picked


def phase_b(n: int, seed: int, out: Path, teacher_answers_path: Path):
    """读 teacher_answers.jsonl → emit distill_sft.jsonl + mixed_sft.jsonl。"""
    # 重建同一批题 (同 seed) → 同 prompt/gold/question_id
    picked = sample_questions(n, seed)
    by_id = {i: q for i, q in enumerate(picked)}

    ans = {}
    for line in teacher_answers_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            ans[a["question_id"]] = a
    print(f"[prepare] teacher_answers 读到 {len(ans)} 条; 题集 {len(picked)}")

    distill_rows = []
    n_compliant = n_recovered_t = n_recovered_g = n_missing = 0
    n_correct = 0
    for i, q in enumerate(picked):
        prompt = fmt_prompt(q)
        gold = q["Answer"].strip().upper()
        a = ans.get(i)
        if a is None:
            n_missing += 1
            continue  # 跳过缺 teacher 答案的题 (distill/mixed 不含)
        comp, letter, status = normalize_distill_completion(a.get("teacher_answer", ""), gold)
        distill_rows.append({"prompt": prompt, "completion": comp, "question_id": i, "gold": gold})
        if status == "compliant":
            n_compliant += 1
        elif status == "recovered_teacher":
            n_recovered_t += 1
        else:
            n_recovered_g += 1
        if letter == gold:
            n_correct += 1

    write_jsonl(out / "distill_sft.jsonl", distill_rows)

    # mixed: N/2 real + N/2 distill (每题一次, 确定性 50/50 分配)
    rng2 = random.Random(seed + 1)
    idx = list(range(len(picked)))
    rng2.shuffle(idx)
    real_half = set(idx[: len(idx) // 2])
    # real completion 查表
    real_comp = {i: real_completion(q) for i, q in enumerate(picked)}
    distill_comp = {r["question_id"]: r["completion"] for r in distill_rows}
    mixed_rows = []
    for i, q in enumerate(picked):
        gold = q["Answer"].strip().upper()
        if i in real_half:
            comp = real_comp[i]
        else:
            if i not in distill_comp:        # 该题 teacher 缺失 → 回退 real, 保 N 不缩
                comp = real_comp[i]
            else:
                comp = distill_comp[i]
        mixed_rows.append({"prompt": fmt_prompt(q), "completion": comp, "question_id": i, "gold": gold})
    rng3 = random.Random(seed + 2)
    rng3.shuffle(mixed_rows)
    write_jsonl(out / "mixed_sft.jsonl", mixed_rows)

    n_eval = len(distill_rows)
    print(f"[prepare] ✓ distill_sft.jsonl ({len(distill_rows)}) + mixed_sft.jsonl ({len(mixed_rows)}) → {out}")
    print(f"[prepare] teacher 合规率 compliant={n_compliant}/{n_eval} "
          f"recovered_teacher={n_recovered_t} recovered_gold={n_recovered_g} missing={n_missing}")
    print(f"[prepare] ★ teacher 对 gold 准确率 = {n_correct}/{n_eval} = {n_correct/max(n_eval,1):.3f} "
          f"(= distill 臂天花板)")
    _sanity(distill_rows, label="distill")


def _sanity(rows: list[dict], label: str, k: int = 2):
    """打印 k 条 (prompt 尾, completion 头) 供人工核对 completion 字母在前。"""
    print(f"[prepare] [{label}] sanity (前 {k} 条 completion 头):")
    for r in rows[:k]:
        print(f"    gold={r['gold']} | completion 头={r['completion'][:60]!r}")


def main():
    p = argparse.ArgumentParser(description="CMExam → 蒸馏三对照 SFT 数据")
    p.add_argument("--n", type=int, default=2000, help="采样题数 (三臂同题集)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=Path, default=Path("phase1/results/week19_distill/data"))
    p.add_argument("--teacher-answers", type=Path, default=None,
                   help="给则进 Phase B (生成 distill+mixed); 不给则只 Phase A (real+teacher_questions)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.teacher_answers:
        phase_b(args.n, args.seed, args.out, args.teacher_answers)
    else:
        phase_a(args.n, args.seed, args.out)


if __name__ == "__main__":
    main()
