"""
Week 5: Loss Masking 手写实现
=============================

SFT 的核心 trick: 只在 assistant response 的 token 上计算 loss,
prompt 部分设为 -100 (PyTorch CrossEntropyLoss 的 ignore_index)。

ChatML 格式:
    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n{user}<|im_end|>\n
    <|im_start|>assistant\n{assistant}<|im_end|>\n

用法:
    python phase0/week5/loss_masking.py
"""

import torch
from transformers import AutoTokenizer


def mask_labels(input_ids, tokenizer, assistant_token_id):
    """
    把 assistant 回复之外的所有 token 的 label 设为 -100。
    只保留 assistant response 部分的 labels, 使训练时只在这些位置计算 loss。

    原理:
        PyTorch CrossEntropyLoss(ignore_index=-100) 会跳过 label=-100 的位置。
        我们把 system prompt + user input 的 label 全部设为 -100,
        只保留 assistant 回复部分的 label 为原始 token id。

    参数:
        input_ids: tokenizer 编码后的 token ids (1D tensor)
        tokenizer: 用于识别 special tokens
        assistant_token_id: "<|im_start|>assistant" 中 assistant 角色对应的 token id

    返回:
        labels: 和 input_ids 同 shape, prompt 位置为 -100, assistant 内容位置保留原值
    """
    labels = input_ids.clone()

    # ChatML special tokens
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Step 1: 默认全部 mask (设为 -100)
    labels[:] = -100

    # Step 2: 找到每个 assistant turn, 只恢复其 content 部分的 labels
    # ChatML 中 assistant turn 的 token 序列:
    #   [<|im_start|>, assistant, \n, ...content..., <|im_end|>]
    # 只保留 content 部分的 labels
    i = 0
    while i < len(input_ids) - 1:
        # 匹配 <|im_start|> + assistant 的模式
        if input_ids[i] == im_start_id and input_ids[i + 1] == assistant_token_id:
            # content 从 <|im_start|>(i) + "assistant"(i+1) 之后开始
            content_start = i + 2

            # 找到对应的 <|im_end|>, 标记 content 结束
            content_end = content_start
            while content_end < len(input_ids) and input_ids[content_end] != im_end_id:
                content_end += 1

            # 恢复 assistant content 部分的 labels (只有这里参与 loss)
            labels[content_start:content_end] = input_ids[content_start:content_end]

            # 跳过已处理的 turn
            i = content_end + 1
        else:
            i += 1

    return labels


def compare_masked_vs_unmasked():
    """对比: masking vs 不 masking 的效果差异"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    # --- 单轮对话 ---
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]

    # 不 masking: 所有 token 都参与 loss 计算
    labels_unmasked = input_ids.clone()

    # masking: 只保留 assistant response 部分
    # 找 assistant 的 token ID (ChatML: <|im_start|> 后紧跟的角色 token)
    assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
    labels_masked = mask_labels(input_ids, tokenizer, assistant_token_id)

    # 统计
    total = len(input_ids)
    masked_count = (labels_masked != -100).sum().item()

    print("=" * 70)
    print("Loss Masking 对比实验 — 单轮")
    print("=" * 70)
    print(f"\n原始 prompt:\n{prompt}")
    print(f"\n{'='*70}")
    print(f"Total tokens:          {total}")
    print(f"不 masking 参与 loss:   {total} (100%)")
    print(f"masking 后参与 loss:    {masked_count} ({100*masked_count/total:.1f}%)")
    print(f"被 mask 掉的 token:     {total - masked_count} ({100*(total-masked_count)/total:.1f}%)")

    # 逐 token 展示 masking 结果
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print(f"\n{'Idx':>4} {'Token':>8} {'Text':>30} {'参与Loss?':>10}")
    print("-" * 60)
    for idx in range(len(input_ids)):
        tok = input_ids[idx].item()
        is_masked = labels_masked[idx].item() != -100
        marker = "✅ loss" if is_masked else "❌ -100"
        # 标记 special tokens
        if tok == im_start_id:
            text = "<|im_start|>"
        elif tok == im_end_id:
            text = "<|im_end|>"
        else:
            text = tokenizer.decode([tok])
        print(f"{idx:>4} {tok:>8} {repr(text):>30} {marker:>10}")

    # --- Multi-turn 测试 ---
    print(f"\n{'='*70}")
    print("Multi-turn 测试 (system + 2 轮对话)")
    print("=" * 70)

    multi_turn = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6."},
    ]

    prompt_mt = tokenizer.apply_chat_template(multi_turn, tokenize=False)
    input_ids_mt = tokenizer.encode(prompt_mt, return_tensors="pt")[0]
    labels_mt = mask_labels(input_ids_mt, tokenizer, assistant_token_id)

    total_mt = len(input_ids_mt)
    masked_mt = (labels_mt != -100).sum().item()

    print(f"Total tokens:          {total_mt}")
    print(f"参与 loss 的 token:     {masked_mt} ({100*masked_mt/total_mt:.1f}%)")
    print(f"被 mask 掉的 token:     {total_mt - masked_mt} ({100*(total_mt-masked_mt)/total_mt:.1f}%)")
    print(f"\n预期: 只有 2 段 assistant 回复参与 loss (约 {masked_mt} tokens)")
    print(f"如果不做 masking, {total_mt} 个 token 全部参与 loss → 模型会浪费梯度学 prompt")


if __name__ == "__main__":
    compare_masked_vs_unmasked()
