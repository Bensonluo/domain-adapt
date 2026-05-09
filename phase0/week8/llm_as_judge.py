"""
Week 8: LLM-as-Judge
====================

实现 pairwise judge: 给定问题 + A/B 两个回答 → LLM 判断哪个更好。

用法:
    python phase0/week8/llm_as_judge.py \
        --questions data/processed/domain_test.jsonl \
        --model_a Qwen/Qwen2.5-3B-Instruct \
        --model_b phase0/week6/domain-sft-merged \
        --output results/week8_judge_results.json

减轻 Judge Bias:
- 位置 bias: swap A/B 跑两次,取一致结果
- 长度 bias: prompt 中提醒不要根据长度判断
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_answer(model, tokenizer, messages, max_new_tokens=256):
    """生成回答"""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def parse_winner(response: str) -> str:
    """从 judge 回复中解析获胜者"""
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("获胜者:") or line.startswith("Winner:"):
            content = line.split(":", 1)[1].strip()
            if "平局" in content or "tie" in content.lower():
                return "tie"
            if "A" in content and "B" not in content:
                return "A"
            if "B" in content and "A" not in content:
                return "B"
            break
    return "tie"


def judge(question, answer_a, answer_b, judge_model, judge_tokenizer):
    """
    用 LLM 作为 judge,判断 A/B 哪个更好。
    使用 chat_template 正确格式化 prompt。
    返回: {"winner": "A"|"B"|"tie", "reason": "..."}
    """
    prompt_text = f"""请客观评估以下两个回答的质量。注意不要根据回答长度来判断优劣。

问题: {question}

回答A:
{answer_a}

回答B:
{answer_b}

请判断哪个回答更好。考虑以下维度:
1. 准确性 (信息是否正确)
2. 完整性 (是否回答了问题的所有方面)
3. 清晰度 (表达是否清楚)

输出格式:
获胜者: A 或 B 或 平局
理由: <简要说明>"""

    messages = [{"role": "user", "content": prompt_text}]
    prompt = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
    response = judge_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    winner = parse_winner(response)
    return {"winner": winner, "reason": response.strip()}


def judge_with_swap(question, answer_a, answer_b, judge_model, judge_tokenizer):
    """
    跑两次 judge (原始 + swap A/B),减轻位置 bias。
    如果两次结果一致 → 返回该结果
    如果不一致 → 返回 tie
    """
    result_1 = judge(question, answer_a, answer_b, judge_model, judge_tokenizer)
    result_2 = judge(question, answer_b, answer_a, judge_model, judge_tokenizer)

    # result_2 中 A/B 的含义是反的,需要翻转
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_2 = swap_map[result_2["winner"]]

    if result_1["winner"] == winner_2:
        return {"winner": result_1["winner"], "reason": result_1["reason"]}
    else:
        return {
            "winner": "tie",
            "reason": f"[位置不一致] 原始: {result_1['winner']}, swap: {winner_2}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="测试问题 JSONL")
    parser.add_argument("--model_a", required=True, help="模型 A")
    parser.add_argument("--model_b", required=True, help="模型 B")
    parser.add_argument(
        "--judge_model", default="Qwen/Qwen2.5-3B-Instruct", help="Judge 模型"
    )
    parser.add_argument("--output", default="phase0/results/week8_judge_results.json")
    parser.add_argument("--max_questions", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    print("加载模型...")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.model_a, device_map=device_map, torch_dtype=dtype
    )
    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b, device_map=device_map, torch_dtype=dtype
    )
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, device_map=device_map, torch_dtype=dtype
    )
    judge_tok = AutoTokenizer.from_pretrained(args.judge_model)

    with open(args.questions, "r") as f:
        questions = [json.loads(line) for line in f][:args.max_questions]

    results = []
    for i, q in enumerate(questions):
        messages = q.get("messages", q)
        user_msg = [m for m in messages if m.get("role") == "user"][-1]
        eval_msg = [{"role": "user", "content": user_msg["content"]}]

        print(f"\n问题 {i+1}/{len(questions)}: {user_msg['content'][:50]}...")
        ans_a = generate_answer(model_a, tok_a, eval_msg)
        ans_b = generate_answer(model_b, tok_b, eval_msg)

        result = judge_with_swap(
            user_msg["content"], ans_a, ans_b, judge_model, judge_tok
        )
        results.append(
            {
                "question": user_msg["content"],
                "answer_a": ans_a,
                "answer_b": ans_b,
                "winner": result["winner"],
                "reason": result["reason"],
            }
        )
        print(f"Judge: {result['winner']}")

    # 统计
    wins_a = sum(1 for r in results if r["winner"] == "A")
    wins_b = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "tie")
    print(f"\n{'='*60}")
    print(f"A 获胜: {wins_a} | B 获胜: {wins_b} | 平局: {ties}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"结果保存: {output}")


if __name__ == "__main__":
    main()
