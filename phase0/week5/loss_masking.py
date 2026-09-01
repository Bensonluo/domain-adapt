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

from transformers import AutoTokenizer


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    """返回 pattern 首次出现的位置；找不到时返回 -1。"""
    if not pattern:
        raise ValueError("pattern 不能为空")
    last = len(sequence) - len(pattern) + 1
    for index in range(start, max(start, last)):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def mask_labels(input_ids, tokenizer):
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
    返回:
        labels: 和 input_ids 同 shape；prompt/角色前缀为 -100，assistant 内容和结束 token 保留原值
    """
    labels = input_ids.clone()

    token_ids = input_ids.tolist()
    assistant_marker = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    turn_end_marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    # 默认全部 mask。
    labels[:] = -100

    # 恢复每个 assistant turn 的正文和 turn-ending token。
    cursor = 0
    turns_found = 0
    while True:
        marker_start = _find_subsequence(token_ids, assistant_marker, cursor)
        if marker_start < 0:
            break
        content_start = marker_start + len(assistant_marker)
        end_start = _find_subsequence(token_ids, turn_end_marker, content_start)
        if end_start < 0:
            raise ValueError(
                "assistant turn 缺少 <|im_end|>；样本可能被截断或 chat template 不匹配"
            )
        supervised_end = end_start + len(turn_end_marker)
        labels[content_start:supervised_end] = input_ids[content_start:supervised_end]
        turns_found += 1
        cursor = supervised_end

    if turns_found == 0:
        raise ValueError(
            "没有找到 assistant marker；拒绝静默退化为全序列 loss，请检查 chat template"
        )

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

    # masking: 只保留 assistant response 部分
    labels_masked = mask_labels(input_ids, tokenizer)

    # 统计
    total = len(input_ids)
    masked_count = (labels_masked != -100).sum().item()

    print("=" * 70)
    print("Loss Masking 对比实验 — 单轮")
    print("=" * 70)
    print(f"\n原始 prompt:\n{prompt}")
    print(f"\n{'=' * 70}")
    print(f"Total tokens:          {total}")
    print(f"不 masking 参与 loss:   {total} (100%)")
    print(f"masking 后参与 loss:    {masked_count} ({100 * masked_count / total:.1f}%)")
    print(
        f"被 mask 掉的 token:     {total - masked_count} ({100 * (total - masked_count) / total:.1f}%)"
    )

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
    print(f"\n{'=' * 70}")
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
    labels_mt = mask_labels(input_ids_mt, tokenizer)

    total_mt = len(input_ids_mt)
    masked_mt = (labels_mt != -100).sum().item()

    print(f"Total tokens:          {total_mt}")
    print(f"参与 loss 的 token:     {masked_mt} ({100 * masked_mt / total_mt:.1f}%)")
    print(
        f"被 mask 掉的 token:     {total_mt - masked_mt} ({100 * (total_mt - masked_mt) / total_mt:.1f}%)"
    )
    print(f"\n预期: 只有 2 段 assistant 回复参与 loss (约 {masked_mt} tokens)")
    print(
        f"如果不做 masking, {total_mt} 个 token 全部参与 loss → 模型会浪费梯度学 prompt"
    )


if __name__ == "__main__":
    compare_masked_vs_unmasked()
