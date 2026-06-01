"""
Phase 1 Week 20: Feature Distillation

对齐 teacher 和 student 的中间层表示。

Usage:
    python phase1/week20/distill_feature.py \
        --teacher Qwen/Qwen2.5-14B \
        --student Qwen/Qwen2.5-3B \
        --output phase1/results/week20_distill_feature/

核心学习目标：
1. 理解 feature distillation 和 response distillation 的本质区别
2. 理解为什么要加 projection layer
3. 思考：中间层表示包含了什么信息？为什么对齐它能提升效果？
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """Feature-level 蒸馏损失：对齐中间层表示"""

    def __init__(self, teacher_hidden_dim: int, student_hidden_dim: int):
        super().__init__()
        self.projection = nn.Linear(student_hidden_dim, teacher_hidden_dim)

    def forward(self, teacher_hidden: torch.Tensor, student_hidden: torch.Tensor) -> torch.Tensor:
        projected_student = self.projection(student_hidden)
        return F.mse_loss(projected_student, teacher_hidden.detach())


class DistillationTrainer:
    """Feature distillation trainer"""

    def __init__(self, teacher, student, tokenizer, layer_map: dict | None = None):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.layer_map = layer_map  # {student_layer_idx: teacher_layer_idx}

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, batch):
        # [YOUR CODE] 单次训练步骤
        # 提示：
        #   1. self.teacher.eval(); self.student.train()
        #   2. teacher forward → 用 hook 或 register_forward_hook 收集 hidden states
        #   3. student forward → 收集对应层的 hidden states
        #   4. 用 FeatureDistillationLoss 计算 MSE loss
        #   5. loss.backward() — 只更新 student（teacher 已 frozen）
        #   6. 返回 loss.item()
        # 思考：为什么要 detach() teacher 的 hidden states？
        pass


def main():
    parser = argparse.ArgumentParser(description="Feature Distillation")
    parser.add_argument("--teacher", required=True, help="Teacher model name or path")
    parser.add_argument("--student", required=True, help="Student model name or path")
    parser.add_argument("--data", default="phase1/data/processed/cpt/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. [YOUR CODE] 加载 teacher + student 模型
    # 提示：
    #   - teacher: 大模型（如 Qwen/Qwen2.5-14B）
    #   - student: 小模型（如 Qwen/Qwen2.5-3B）
    #   - teacher 加载后设置 eval() + requires_grad=False
    #   - 思考：feature distillation 要求 teacher 和 student 架构相似吗？
    teacher = None  # TODO: 加载 teacher
    student = None  # TODO: 加载 student
    tokenizer = None  # TODO: 加载 tokenizer

    # 2. [YOUR CODE] 自动检测 hidden dims + 创建 layer map
    # 提示：
    #   - 比较 teacher.config.hidden_size 和 student.config.hidden_size
    #   - 如果不同，FeatureDistillationLoss 会自动创建 projection
    #   - layer_map: 决定 student 的哪层对齐 teacher 的哪层
    #   - 思考：layer_map 怎么设计最合理？均匀映射还是最后一层？
    teacher_hidden_dim = None  # TODO: teacher.config.hidden_size
    student_hidden_dim = None  # TODO: student.config.hidden_size
    layer_map = None  # TODO: 设计 layer 映射

    # 3. 创建 FeatureDistillationLoss（已实现）
    distill_loss_fn = FeatureDistillationLoss(teacher_hidden_dim, student_hidden_dim)

    # 4. [YOUR CODE] 加载数据 + 构建训练循环
    # 提示：
    #   - 加载 CPT 语料（和 week11 同样的数据）
    #   - 用标准 PyTorch DataLoader
    #   - 每轮：batch → train_step() → 记录 loss
    #   - 思考：feature distillation 的 loss 和 next-token prediction loss 怎么组合？
    dataset = None  # TODO: 加载数据

    # 5. [YOUR CODE] 训练 + 评估
    # trainer = DistillationTrainer(teacher, student, tokenizer, layer_map)
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         loss = trainer.train_step(batch)
    #         print(f"loss: {loss:.4f}")
    print("TODO: Implement training loop")

    # 6. [YOUR CODE] 保存 student 模型
    # student.save_pretrained(args.output)
    print(f"TODO: Save distilled model to {args.output}")


if __name__ == "__main__":
    main()
