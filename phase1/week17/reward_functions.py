"""
Phase 1 Week 17: Reward Functions

Domain-specific reward functions for GRPO training.

设计方案:
A. Rule-based: 格式正确性 + 关键词覆盖 + 长度惩罚
B. Model-based: 用另一个 LLM 做 answer quality 评分
C. Hybrid: A + B 加权组合
"""

from typing import Callable


def rule_based_reward(question: str, answer: str) -> float:
    """基于规则的 reward function"""
    score = 0.0

    # 1. 格式检查（0.3 分）
    if len(answer) > 50:
        score += 0.3

    # 2. 关键词覆盖（0.4 分）
    # TODO: 实现 extract_medical_terms
    # medical_terms = extract_medical_terms(question)
    # covered = sum(1 for t in medical_terms if t in answer)
    # score += 0.4 * (covered / max(len(medical_terms), 1))
    score += 0.2  # placeholder

    # 3. 长度惩罚（0.3 分）— 避免过短或过长
    if 100 < len(answer) < 500:
        score += 0.3

    return score


def model_based_reward(question: str, answer: str, judge_model: str = "gpt-4o") -> float:
    """用 LLM 做 answer quality 评分"""
    # TODO: 实现 LLM 评分（需要先实现 utils/call_llm.py）
    # prompt = f"Rate this answer 1-5 on accuracy and completeness.\nQ: {question}\nA: {answer}"
    # score = call_llm(judge_model, prompt)
    # return float(score) / 5.0
    raise NotImplementedError(
        "model_based_reward 需要先实现 utils/call_llm.py — "
        "详见 week17/README.md 的 reward function 设计"
    )


def hybrid_reward(
    rule_weight: float = 0.4,
    model_weight: float = 0.6,
) -> Callable[[str, str], float]:
    """混合 reward: rule-based + model-based"""

    def _reward(question: str, answer: str) -> float:
        rule_score = rule_based_reward(question, answer)
        model_score = model_based_reward(question, answer)
        return rule_weight * rule_score + model_weight * model_score

    return _reward


# Registry for GRPOTrainer
REWARD_FUNCTIONS = {
    "rule_based": rule_based_reward,
    "model_based": model_based_reward,
    "hybrid": hybrid_reward(),
}
