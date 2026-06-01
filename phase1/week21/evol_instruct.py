"""
Phase 1 Week 21: Evol-Instruct Pipeline

简单问题 → 大模型改写成更复杂的问题。

Usage:
    python phase1/week21/evol_instruct.py \
        --input phase1/data/processed/synthetic/self_instruct.jsonl \
        --model gpt-4o \
        --output phase1/data/processed/synthetic/evolved/
"""

import argparse


EVOL_PROMPTS = {
    "constraints": "Add more constraints or requirements to this question.",
    "deepening": "Increase the reasoning depth required to answer this question.",
    "concretizing": "Make this question more specific and concrete.",
    "reasoning": "Rewrite to require step-by-step reasoning.",
}


def evolve_question(seed_question: str, strategy: str = "deepening", model: str = "gpt-4o") -> str:
    """用大模型演化问题复杂度"""
    # TODO: 实现 LLM 调用
    instruction = EVOL_PROMPTS.get(strategy, EVOL_PROMPTS["deepening"])
    # prompt = f"Given this question, generate a MORE COMPLEX version:\n{instruction}\n\nSeed: {seed_question}\n\nEvolved:"
    # return call_llm(model, prompt)
    return seed_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input instructions JSONL")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--depth", type=int, default=1, help="Evolution depth (1-3)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # TODO: 加载输入指令
    # TODO: 多步演化（depth 轮）
    # TODO: 每轮随机选择演化策略
    # TODO: 质量过滤
    # TODO: 保存演化后数据

    print("TODO: Implement evol-instruct pipeline")


if __name__ == "__main__":
    main()
