"""
Week 6: 人工评估
==============

对 base model 和 finetuned model 进行人工对比评估。

用法:
    python phase0/week6/eval_manual.py \
        --base_model Qwen/Qwen2.5-3B-Instruct \
        --finetuned_model ./domain-sft-merged \
        --questions data/processed/domain_test.jsonl \
        --output results/week6_manual_eval.md
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_questions(path: str):
    """加载评估问题"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_answer(model, tokenizer, messages, max_new_tokens=256):
    """生成回答"""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_models(base_model_path: str, finetuned_model_path: str, questions_path: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("加载 base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto" if device == "cuda" else None, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print("加载 finetuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, device_map="auto" if device == "cuda" else None, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)

    questions = load_questions(questions_path)
    print(f"评估问题数: {len(questions)}")

    results = []
    for i, q in enumerate(questions[:20]):  # 先评估 20 题
        messages = q.get("messages", q)
        # 提取 user 问题
        user_msg = [m for m in messages if m.get("role") == "user"][-1]
        eval_messages = [{"role": "user", "content": user_msg["content"]}]

        base_answer = generate_answer(base_model, base_tokenizer, eval_messages)
        ft_answer = generate_answer(ft_model, ft_tokenizer, eval_messages)

        results.append({
            "id": i,
            "question": user_msg["content"],
            "base_answer": base_answer,
            "ft_answer": ft_answer,
        })
        print(f"\n--- 问题 {i+1} ---")
        print(f"Q: {user_msg['content'][:100]}...")
        print(f"Base: {base_answer[:100]}...")
        print(f"FT: {ft_answer[:100]}...")

    # 保存结果
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("# Week 6 人工评估记录\n\n")
        f.write("评分标准: 1-5 分\n")
        f.write("- 1分: 完全错误/无关\n")
        f.write("- 2分: 部分相关但有明显错误\n")
        f.write("- 3分: 基本正确但不完整\n")
        f.write("- 4分: 正确且较完整\n")
        f.write("- 5分: 准确、完整、有深度\n\n")
        f.write("| 题号 | 问题 | Base 评分 | FT 评分 | 备注 |\n")
        f.write("|------|------|-----------|---------|------|\n")
        for r in results:
            f.write(f"| {r['id']} | {r['question'][:50]}... | | | |\n")
        f.write("\n## 详细回答\n\n")
        for r in results:
            f.write(f"### 问题 {r['id']}: {r['question']}\n\n")
            f.write(f"**Base:**\n{r['base_answer']}\n\n")
            f.write(f"**Finetuned:**\n{r['ft_answer']}\n\n---\n\n")

    print(f"\n评估结果已保存: {output}")
    print("请打开文件,为每个回答打分。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--finetuned_model", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="phase0/results/week6_manual_eval.md")
    args = parser.parse_args()
    eval_models(args.base_model, args.finetuned_model, args.questions, args.output)
