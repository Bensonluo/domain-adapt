"""
Phase1 Week20 Part A: teacher 离线 logits 提取 (mlx_lm forward, 不 generate)

读 week19 distill_sft.jsonl (prompt + completion + question_id), teacher MLX forward
(prompt_ids ⊕ completion_ids), 取 completion 区域每位置 top-K raw logits + token_id,
存 teacher_topk_logits.jsonl. 给 train_logit_kd.py 做 restricted-KL on top-K.

与 week19 distill 臂完全同题同 completion → 干净对照 hard(label) vs soft(logits) label.

★ 设计要点:
  - **raw prompt + completion** (student 视角), 不走 chat template → teacher/student 同输入,
    KD token 级对齐才成立 (teacher 用 teacher_user_msg 会与 student context 错位).
  - causal LM shift: completion token c_t 在 input position L+t, 由 logits[L-1+t] 预测
    (L=prompt_len) → completion 区域 = logits[:, L-1 : L+M-1, :], M 个位置.
  - 存 **raw logits** (不 logsoftmax), 温度 T 训练时统一施加 → 多温度变体复用同一份 logits.
  - top-K (默认 20): restricted-KL 标准折中, 存储友好 (2000×~50×20×8B ≈ 16MB).
  - sanity: completion 首位 (字母) 的 teacher top-1 应 == 字母 (否则 raw prompt 让 teacher 异常).

环境: 4bit-QLoRA-post-training venv (有 mlx_lm 0.31.3). Apple Silicon → MTL_TIMEOUT=0.

Usage:
  # SMOKE (50 题, ~1min, 验证 logits shape + top1 sanity)
  VENV=/Users/luopeng/Documents/GitHub/4bit-QLoRA-post-training/venv/bin/python
  $VENV phase1/week20/extract_teacher_logits.py \\
    --teacher ~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit \\
    --data phase1/results/week19_distill/data/distill_sft.jsonl \\
    --out phase1/results/week20_distill/data/teacher_topk_logits.jsonl --limit 50
  # FULL (--resume)
  $VENV phase1/week20/extract_teacher_logits.py ... --resume
"""

import argparse
import json
import os
import time
from pathlib import Path

# Apple Silicon Metal 超时禁 (week11 CPT / week19 teacher 同款坑)
os.environ.setdefault("MTL_TIMEOUT", "0")


def main():
    p = argparse.ArgumentParser(description="teacher 离线 logits 提取 (mlx_lm forward)")
    p.add_argument("--teacher", required=True, help="MLX teacher 模型目录")
    p.add_argument("--data", required=True, help="week19 distill_sft.jsonl")
    p.add_argument("--out", required=True, help="teacher_topk_logits.jsonl (append+resume)")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="smoke: 只跑前 N (0=全部)")
    p.add_argument("--resume", action="store_true", help="跳过 out 里已有的 question_id")
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    import mlx.core as mx
    import mlx_lm
    import numpy as np

    t0 = time.time()
    model, tokenizer = mlx_lm.load(args.teacher)
    print(f"[logits] 加载 teacher {args.teacher} ({time.time()-t0:.1f}s)", flush=True)

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    done = set()
    if args.resume and Path(args.out).exists():
        for line in Path(args.out).read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])
    todo = [r for r in rows if r["question_id"] not in done]
    print(f"[logits] 总 {len(rows)} | 已完成 {len(done)} (resume) | todo {len(todo)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    K = args.topk
    n_done = 0
    n_top1_letter = 0          # sanity: completion 首位 top-1 应是字母
    n_top1_match_gold = 0      # 额外: top-1 首位 == gold?
    tot_comp_tok = 0
    t_start = time.time()

    # 字母 token 集合 (A-E 单字符); tokenizer encode("A")=[32] (week19 sanity 已确认)
    letter_tokens = set()
    for ch in "ABCDE":
        letter_tokens.update(tokenizer.encode(ch))

    with open(args.out, "a", encoding="utf-8") as fout:
        for r in todo:
            qid = r["question_id"]
            gold = r.get("gold", "")
            prompt_ids = tokenizer.encode(r["prompt"])
            comp_ids = tokenizer.encode(r["completion"])
            L, M = len(prompt_ids), len(comp_ids)
            if L == 0 or M == 0:
                print(f"[logits] ⚠ qid={qid} 空 prompt/completion, 跳过", flush=True)
                continue
            # forward (不传 cache: 一次过拿全部位置, 不要 KV-cache 增量)
            input_ids = mx.array(prompt_ids + comp_ids)[None, :]   # (1, L+M)
            logits = model(input_ids)                              # (1, L+M, V) lazy
            # completion 区域 (causal shift): positions L-1 .. L+M-2, 共 M 个
            comp_logits = logits[0, L - 1 : L + M - 1, :]          # (M, V)
            # mlx → numpy: 量化模型 forward 返回 bf16, 先 mlx 内转 f32 再转 np
            # (np.array(copy=False) 走 PEP3118 buffer 撞 bf16 itemsize 不匹配)
            arr = np.array(comp_logits.astype(mx.float32))        # (M, V) float32
            # 每行 top-K: argpartition 选 top-K (O(V)), 再降序排
            part_idx = np.argpartition(-arr, K - 1, axis=-1)[:, :K]      # (M, K) 未排序
            part_logits = arr[np.arange(M)[:, None], part_idx]          # (M, K) 对应 logits
            order = np.argsort(-part_logits, axis=-1)                   # (M, K) 降序序
            top_idx = np.take_along_axis(part_idx, order, axis=-1)      # (M, K) 降序 token ids
            top_logits = np.take_along_axis(part_logits, order, axis=-1)  # (M, K) 降序 logits

            positions = [{
                "pos": t,
                "topk_tokens": [int(x) for x in top_idx[t]],
                "topk_logits": [float(x) for x in top_logits[t]],
            } for t in range(M)]

            rec = {
                "question_id": qid,
                "gold": gold,
                "prompt_len": L,
                "completion_token_ids": [int(x) for x in comp_ids],
                "positions": positions,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            n_done += 1
            tot_comp_tok += M
            # sanity: completion 首位 top-1 token decode 是字母?
            first_top1 = int(top_idx[0, 0])
            first_tok_str = tokenizer.decode([first_top1]).strip()
            if first_tok_str and first_tok_str[0].upper() in "ABCDE":
                n_top1_letter += 1
                if first_tok_str[0].upper() == gold.upper():
                    n_top1_match_gold += 1

            if n_done % args.log_every == 0 or n_done == len(todo):
                wall = time.time() - t_start
                print(f"[logits] {n_done}/{len(todo)} | comp_tok={tot_comp_tok} | "
                      f"top1_字母={n_top1_letter}/{n_done} ({n_top1_letter/n_done:.2f}) | "
                      f"top1==gold={n_top1_match_gold}/{n_done} ({n_top1_match_gold/n_done:.2f}) | "
                      f"wall={wall:.0f}s", flush=True)

    wall = time.time() - t_start
    print(f"\n[logits] ✓ 本次 {n_done} 条 → {args.out}", flush=True)
    if n_done:
        print(f"[logits] tot_comp_tok={tot_comp_tok} (均值 {tot_comp_tok/n_done:.1f}/条) | "
              f"★ top1_字母={n_top1_letter/n_done:.3f} | top1==gold={n_top1_match_gold/n_done:.3f} | "
              f"wall={wall:.0f}s", flush=True)
        if n_top1_letter / n_done < 0.9:
            print("[logits] ⚠⚠ top1 字母率 < 0.9 → raw prompt 可能让 teacher 异常, "
                  "需检查是否该用 chat template (会牺牲 student/teacher 同输入对齐)", flush=True)


if __name__ == "__main__":
    main()
