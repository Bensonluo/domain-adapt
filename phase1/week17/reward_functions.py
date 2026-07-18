"""
Phase 1 Week 17: Reward Functions for GRPO (MCQ 答对率)

设计 (见 week17 plan): MCQ 客观 reward — 解析模型 completion 提取答案字母 (A-E),
与标准答案比对, 逐条 1.0/0.0。

为什么 MCQ (不是开放式 rule-based):
  开放式医疗 reward 区分度弱 (G 个 completion 格式近似 → reward 同分 → advantage≈0 → 学不动)
  且易 reward hacking。MCQ 客观、区分度天然、hacking 几乎不可能, GRPO 机制
  (on-policy / group baseline / KL 漂移) 能干净观察。学习价值最大。

TRL v1.8.0 reward 签名 (grpo_trainer.py L1513-1520 关键字调用):
    def f(prompts, completions, completion_ids, **kwargs) -> list[float]
  - 关键字调用, 不是位置参数 (现有旧 stub 的 (question, answer)->float 会崩)
  - 返回逐条 list (长 = B×G), 不是单 float
  - 额外 dataset 列 (如 answer) 按列名进 kwargs (需 GRPOConfig.remove_unused_columns=False)
"""

import re
from typing import Callable

# 答案字母正则。我们的 prompt 以 "答案：" 结尾 → 模型 completion 一般**首字符**即答案字母
# (如 "D\n\n解析：...")。故解析优先级:
#   1) 首字符即 A-E (允许前导空格/全半角括号)  — prompt 设计下的主信号, 最可靠
#   2) 显式标记 "答案是X"/"正确答案为X"/"选择X"   — 模型写了推理才用 (排除 "选项X" — 那是引用非断言)
#   3) 文中首个独立 A-E                           — 兜底
_LEAD = re.compile(r"^\s*[（(]?\s*([A-Ea-e])\b")          # 首字符即字母 (允许 (X) / （X）)
_EXPLICIT = re.compile(r"(?:答案|正确答案|选择)[是为：:\s]*([A-Ea-e])")
_ANY = re.compile(r"\b([A-Ea-e])\b")


def extract_answer(text: str) -> str | None:
    """从 completion 提取答案字母 (大写)。优先级: 首字符 → 显式标记 → 首个字母 → None。"""
    if not text:
        return None
    m = _LEAD.search(text)
    if m:
        return m.group(1).upper()
    m = _EXPLICIT.search(text)
    if m:
        return m.group(1).upper()
    m = _ANY.search(text)
    if m:
        return m.group(1).upper()
    return None


def to_text(completion) -> str:
    """兼容 str 和 chat list[dict] 两种格式 (TRL is_conversational 自动判, L1563)。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


def mcq_accuracy_reward(
    prompts,            # list[str] | list[list[dict]], 长 = B×G
    completions,        # 同上, 与 prompts 逐条对齐
    answer,             # list[str], 来自 dataset "answer" 列
    **kwargs,           # 吸收 completion_ids / trainer_state / log_extra 等
) -> list[float]:
    """MCQ 答对率 reward: 解析 completion 答案字母 vs 标准 answer, 逐条 1.0/0.0。

    客观 reward → 区分度天然 (G 个 completion 有对有错 → std>0 → advantage 非 0),
    reward hacking 几乎不可能 (答对就是答对)。
    """
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_answer(to_text(comp))
        gold = (gold or "").strip().upper()
        gold_letter = gold[0] if gold else ""   # gold 可能是 "B" 或 "B.xxx", 取首字母
        rewards.append(1.0 if pred is not None and pred == gold_letter else 0.0)
    return rewards


def rule_based_format_reward(
    prompts,
    completions,
    **kwargs,
) -> list[float]:
    """[备用 / 非主线] 纯格式 reward: 长度区间 + 是否含可解析答案字母。

    week14 矩阵行 5 reward ablation 用 (去内容信号只看格式 → 验 reward hacking 是否出现)。
    不含医学关键词覆盖 (需词典 + 非主线, 避免多余工程)。
    """
    rewards = []
    for comp in completions:
        text = to_text(comp)
        score = 0.0
        if 20 < len(text) < 500:
            score += 0.5
        if extract_answer(text) is not None:
            score += 0.5
        rewards.append(score)
    return rewards


# Registry: GRPOTrainer reward_funcs 接单个 callable 或 list[callable]
# (model_based 砍掉 — LLM judge 付费 + 外发数据, 违反隐私约束)
REWARD_FUNCTIONS = {
    "mcq_accuracy": mcq_accuracy_reward,   # 主线: 客观答对率
    "format": rule_based_format_reward,    # 备用: 格式 (ablation)
}

DEFAULT_REWARD = "mcq_accuracy"
