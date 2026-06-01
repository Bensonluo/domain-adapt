"""
Phase 1 Week 17: GRPO 训练脚本

用 TRL GRPOTrainer 做 Group Relative Policy Optimization。

Usage:
    python phase1/week17/train_grpo.py \
        --model phase1/results/week11_cpt_pure/ \
        --reward rule_based \
        --output phase1/results/week17_grpo_rule/

核心学习目标：
1. 理解 GRPO 为什么不需要 reference model 和显式 reward model
2. 设计 domain-specific reward function（这是 GRPO 的核心）
3. 识别和诊断 reward hacking 现象
4. 对比 GRPO vs DPO 在相同数据上的效果差异
"""

import argparse
import os
import sys
from pathlib import Path

# --- [YOUR CODE] 导入必要的库 ---
# 提示：需要 trl, transformers, datasets
# from trl import GRPOConfig, GRPOTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset, load_from_disk
# import torch

# 从 week17 导入 reward functions
# sys.path.insert(0, str(Path(__file__).parent))
# from reward_functions import REWARD_FUNCTIONS


# --- 辅助工具（已提供） ---
def get_device() -> str:
    """检测可用的计算设备"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="GRPO 训练")
    parser.add_argument("--model", required=True, help="SFT model path")
    parser.add_argument("--data", default="phase1/data/processed/sft/", help="Prompt dataset directory")
    parser.add_argument("--reward", default="rule_based", choices=["rule_based", "model_based", "hybrid"])
    parser.add_argument("--num-generations", type=int, default=8, help="Generations per prompt")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--kl-coeff", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. 设备检测
    device = get_device()
    print(f"Using device: {device}")

    # 2. [YOUR CODE] 加载 model + tokenizer
    # 提示：
    #   - 加载 args.model 路径的模型（CPT+SFT 后的模型）
    #   - 思考：GRPO 不需要 reference model，为什么？
    model = None  # TODO: AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = None  # TODO: AutoTokenizer.from_pretrained(args.model)

    # 3. [YOUR CODE] 加载 reward functions
    # 提示：
    #   - 从 week17/reward_functions.py 导入 REWARD_FUNCTIONS
    #   - 根据 args.reward 选择：
    #     "rule_based" → rule_based_reward（已实现，可用）
    #     "model_based" → model_based_reward（需先实现 utils/call_llm.py）
    #     "hybrid" → hybrid_reward()（rule + model 加权）
    #   - 注意：GRPOTrainer 的 reward_funcs 参数接受 list[Callable]
    #   - 思考：为什么 GRPO 可以接收多个 reward function？
    reward_funcs = []  # TODO: [REWARD_FUNCTIONS[args.reward]]

    # 4. [YOUR CODE] 加载 prompt 数据集
    # 提示：
    #   - GRPO 只需要 prompts（不需要完整偏好对）
    #   - 格式: [{"prompt": str}, ...] 或 datasets.Dataset
    #   - 思考：GRPO 和 DPO 在数据需求上有什么不同？
    dataset = None  # TODO: 加载 prompt 数据

    # 5. [YOUR CODE] 配置 GRPOConfig
    # 提示：
    #   - 参考 week17/README.md
    #   - 关键参数：
    #     num_generations=args.num_generations（建议对比 4/8/16）
    #     temperature=args.temperature
    #     learning_rate=args.lr
    #     per_device_train_batch_size=1（GRPO 显存消耗大）
    #   - 思考：num_generations 从 4 增加到 16，效果一定变好吗？为什么？
    grpo_config = None  # TODO: GRPOConfig(output_dir=args.output, ...)

    # 6. [YOUR CODE] 创建 GRPOTrainer
    # 提示：
    #   - model, reward_funcs, args, train_dataset, processing_class=tokenizer
    #   - 注意：reward_funcs 是 list[Callable]，可以传多个
    trainer = None  # TODO: GRPOTrainer(model=model, reward_funcs=reward_funcs, ...)

    # 7. [YOUR CODE] 训练 + 监控 reward hacking
    # 提示：
    #   - trainer.train() 启动
    #   - 每 100-200 步：抽样 10 条生成，人工评分
    #   - 如果 reward 上升但人工评分下降 → reward hacking 出现
    #   - 思考：怎么区分"真实提升"和"reward hacking"？
    # trainer.train()
    # trainer.save_model(args.output)
    print("TODO: Implement GRPO training")
    print(f"  Reward: {args.reward}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  KL coeff: {args.kl_coeff}")

    # 8. [YOUR CODE] 保存训练配置 + reward hacking 分析记录
    # 提示：记录超参、reward 曲线、抽样评估结果


if __name__ == "__main__":
    main()
