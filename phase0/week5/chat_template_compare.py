"""
Week 5: Chat Template 对比实验
==============================

用同一条对话,分别用 Qwen/ChatML、Llama-3、Mistral 的 template 进行 tokenize,
对比 token 数、特殊 token 分布、template 错配的后果。

用法:
    python phase0/week5/chat_template_compare.py
"""

from transformers import AutoTokenizer


CONVERSATION = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "What are the symptoms of diabetes?"},
    {"role": "assistant", "content": "Common symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, and blurred vision."},
]


def compare_templates():
    models = {
        "Qwen2.5": "Qwen/Qwen2.5-3B-Instruct",
        "Llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    }

    for name, model_id in models.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # 用模型自带的 chat_template
            prompt = tokenizer.apply_chat_template(
                CONVERSATION, tokenize=False, add_generation_prompt=True
            )
            tokens = tokenizer.encode(prompt)
            print(f"\n{'='*60}")
            print(f"Model: {name} ({model_id})")
            print(f"Token count: {len(tokens)}")
            print(f"Template preview:\n{prompt[:300]}...")
        except Exception as e:
            print(f"\n{name}: 加载失败 ({e})")

    # TODO: 额外实验
    # 1. 用 Llama-3 template 去 tokenize Qwen 的对话格式,观察差异
    # 2. 统计 system/user/assistant 各角色占多少 token
    # 3. 如果 template 错配(比如用 Mistral template 训练 Qwen 模型),推理时会怎样?


if __name__ == "__main__":
    compare_templates()
    print("\n\n提示: 如果某些模型需要 HuggingFace 登录,请先运行 `huggingface-cli login`")
