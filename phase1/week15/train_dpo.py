"""
Phase 1 Week 15: DPO 训练脚本

用 TRL DPOTrainer 做 Direct Preference Optimization。

Usage:
    python phase1/week15/train_dpo.py \
        --model phase1/results/week11_cpt_pure/ \
        --data phase1/data/processed/preference/ \
        --beta 0.3 \
        --output phase1/results/week15_dpo_0.3/

核心学习目标：
1. 理解 DPO 为什么不需要显式训练 reward model
2. 观察 chosen/rejected reward margin 的变化趋势
3. 理解 beta 参数如何控制对齐强度
"""

import argparse
import os

# --- [YOUR CODE] 导入必要的库 ---
# 提示：需要 trl, transformers, datasets
# from trl import DPOConfig, DPOTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from datasets import load_dataset, load_from_disk
# import torch


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
    parser = argparse.ArgumentParser(description="DPO 训练")
    parser.add_argument("--model", required=True, help="SFT model path")
    parser.add_argument("--data", default="phase1/data/processed/preference/", help="Preference data directory")
    parser.add_argument("--beta", type=float, default=0.3, help="DPO beta (alignment strength)")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (smaller than SFT)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0, help="Flip this fraction of preferences (failure mode experiment)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. 设备检测
    device = get_device()
    print(f"Using device: {device}")

    # 2. [YOUR CODE] 加载 model + reference model
    # 提示：
    #   - policy model: 加载 args.model 路径的模型（CPT+SFT 后的模型）
    #   - reference model: 通常和 policy 同架构，但 frozen（requires_grad=False）
    #   - 思考：为什么 DPO 需要 reference model？它的作用是什么？
    model = None  # TODO: AutoModelForCausalLM.from_pretrained(args.model)
    ref_model = None  # TODO: AutoModelForCausalLM.from_pretrained(args.model) 然后 frozen
    tokenizer = None  # TODO: AutoTokenizer.from_pretrained(args.model)

    # 3. [YOUR CODE] 加载偏好数据集
    # 提示：
    #   - 格式: {"prompt": str, "chosen": str, "rejected": str}
    #   - 可以用 datasets.load_from_disk() 或 load_dataset("json", data_files=...)
    #   - DPOTrainer 需要数据被 tokenizer 处理过（或传 processing_class=tokenizer）
    #   - 思考：preference data 和 SFT data 在格式上有什么区别？
    dataset = None  # TODO: 加载偏好数据

    # 4. [YOUR CODE] 如果 --noise > 0，翻转部分偏好标签（搞坏实验）
    # 提示：
    #   - 随机选择 args.noise 比例的数据对
    #   - 交换 chosen 和 rejected
    #   - 记录哪些对被翻转了（方便分析）
    #   - 思考：如果 30% 的偏好标签是错的，DPO 会崩溃到什么程度？
    if args.noise > 0:
        # TODO: 实现噪声注入
        print(f"Injecting {args.noise * 100:.0f}% noise into preference labels")
        pass

    # 5. [YOUR CODE] 配置 DPOConfig
    # 提示：
    #   - 参考 week15/README.md
    #   - 关键参数：
    #     beta=args.beta（对齐强度，建议对比 0.1/0.3/0.5）
    #     learning_rate=args.lr（比 SFT 小，通常 5e-7）
    #     num_train_epochs=args.epochs
    #     bf16=True
    #   - 思考：beta 太大或太小各有什么后果？
    dpo_config = None  # TODO: DPOConfig(output_dir=args.output, beta=args.beta, ...)

    # 6. [YOUR CODE] 创建 DPOTrainer
    # 提示：
    #   - model, ref_model, args, train_dataset, processing_class=tokenizer
    #   - 注意：processing_class 是 TRL 0.12+ 的参数名（旧版叫 tokenizer）
    trainer = None  # TODO: DPOTrainer(model=model, ref_model=ref_model, args=dpo_config, ...)

    # 7. [YOUR CODE] 训练 + 监控 reward margin
    # 提示：
    #   - trainer.train() 启动
    #   - 监控 chosen/rejected reward margin（可以通过回调或 wandb）
    #   - 思考：reward margin 应该越来越大还是趋于稳定？
    # trainer.train()
    # trainer.save_model(args.output)
    print("TODO: Implement DPO training")
    print(f"  Beta: {args.beta}")
    print(f"  LR: {args.lr}")
    print(f"  Noise: {args.noise}")


if __name__ == "__main__":
    main()
