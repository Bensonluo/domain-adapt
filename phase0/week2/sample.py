"""
Week 2: 生成文本样例
==================

从训练好的 nanoGPT checkpoint 生成文本,支持 temperature 和 top-k。

用法:
    python phase0/week2/sample.py --checkpoint phase0/checkpoints/nanogpt_domain/best.pt --prompt "患者主诉" --temperature 0.8
"""

import argparse
import sys
from pathlib import Path

import torch
import tiktoken

# 确保从 repo 根目录也能正确 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_nanogpt import GPT, GPTConfig


def generate(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    config = GPTConfig()
    config.block_size = args.block_size
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.n_embd = args.n_embd

    model = GPT(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt if isinstance(ckpt, dict) and "model_state" not in ckpt else ckpt.get("model_state", ckpt))
    model.eval()

    start_ids = enc.encode(args.prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    text = enc.decode(y[0].tolist())
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="\n", help="生成起始文本")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    args = parser.parse_args()
    generate(args)
