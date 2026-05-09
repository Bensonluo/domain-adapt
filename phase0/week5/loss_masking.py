"""
Week 5: Loss Masking 手写实现
=============================

SFT 的核心 trick: 只在 assistant response 的 token 上计算 loss,
prompt 部分设为 -100 (PyTorch CrossEntropyLoss 的 ignore_index)。

用法:
    python phase0/week5/loss_masking.py
"""

import torch
from transformers import AutoTokenizer


def mask_labels(input_ids, tokenizer, assistant_token_id):
    """
    把 assistant 回复之前的所有 token 的 label 设为 -100。

    参数:
        input_ids: tokenizer 编码后的 token ids
        tokenizer: 用于识别 special tokens
        assistant_token_id: assistant 角色开始的 token id

    返回:
        labels: 和 input_ids 同 shape, prompt 位置为 -100
    """
    labels = input_ids.clone()

    # TODO: 实现 masking 逻辑
    # 提示:
    #   1. 找到所有 assistant_token_id 的位置
    #   2. 对每个 assistant turn:
    #      - 从 assistant_token_id 开始到下一个 special token 结束
    #      - 把这些位置的 labels 保留原值
    #      - 其他位置设为 -100
    #   3. 最后一个 assistant turn 之后的 eos/im_end token 也设为 -100

    raise NotImplementedError("实现 mask_labels")
    return labels


def compare_masked_vs_unmasked():
    """对比: masking vs 不 masking 的效果差异"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]

    # 不 masking: 所有 token 都参与 loss 计算
    labels_unmasked = input_ids.clone()

    # masking: 只保留 assistant response 部分
    # TODO: 找到 assistant 开始的 token id
    # labels_masked = mask_labels(input_ids, tokenizer, assistant_token_id=...)

    print(f"Prompt token count: {len(input_ids)}")
    print(f"不 masking 时参与 loss 的 token: {len(input_ids)}")
    # print(f"masking 后参与 loss 的 token: {(labels_masked != -100).sum().item()}")

    # TODO: 跑一个 toy 训练实验:
    #   1. 不 masking 训练 100 步
    #   2. masking 训练 100 步
    #   3. 对比两者的生成效果
    #   预期: 不 masking 的模型会学会"重复用户问题"


if __name__ == "__main__":
    compare_masked_vs_unmasked()
