"""
Week 1 训练循环: 在 tiny_shakespeare 上训 MiniGPT
================================================

最小可跑训练循环。Mac MPS / CPU / CUDA 都能跑(自动检测设备)。

参数 (tiny 配置,Mac CPU 5-15 分钟):
    model_dim=128, n_heads=4, n_layers=4, block_size=128
    batch_size=32, max_iters=2000

数据:
    自动下载 tiny_shakespeare (~1MB) 到 phase0/data/raw/

跑法:
    python phase0/week1/train_toy.py

产物:
    phase0/checkpoints/mini_gpt_toy.pt    最终权重
    phase0/results/loss_curve.png         loss 曲线
    phase0/results/sample.txt             生成样例

验收点:
    [ ] loss 从 ~4.5 (随机) 降到 < 2.0
    [ ] 生成的文本有"看起来像莎士比亚"的形态(虽然语义可能乱)
    [ ] 截图保存到 phase0/results/ 作为交付物
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import torch

# 让 phase0/week1/*.py 能 import 同目录下的 day5_7_transformer
sys.path.insert(0, str(Path(__file__).parent))

from day5_7_transformer import MiniGPT  # noqa: E402

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "phase0" / "data" / "raw"
CKPT_DIR = ROOT / "phase0" / "checkpoints"
RESULTS_DIR = ROOT / "phase0" / "results"

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = DATA_DIR / "tiny_shakespeare.txt"

# 超参 (Mac CPU 友好)
CONFIG = {
    "block_size": 128,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "dropout": 0.1,
    "batch_size": 32,
    "max_iters": 2000,
    "eval_interval": 200,
    "eval_iters": 50,
    "lr": 3e-4,
    "warmup_iters": 100,
}


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def download_data() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        print(f"下载 tiny_shakespeare 到 {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    text = DATA_FILE.read_text(encoding="utf-8")
    print(f"语料长度: {len(text):,} 字符")
    return text


def make_char_tokenizer(text: str) -> tuple[dict[str, int], dict[int, str], int]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机取 batch_size 个起点,各取 block_size+1 个字符做训练对"""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(step: int, base_lr: float, warmup: int) -> float:
    """简单 warmup,之后保持常数(教学版)"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def main() -> None:
    device = select_device()
    print(f"device = {device}")

    text = download_data()
    stoi, itos, vocab_size = make_char_tokenizer(text)
    print(f"vocab_size = {vocab_size}")
    data = encode(text, stoi)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        block_size=CONFIG["block_size"],
        dropout=CONFIG["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    history = {"step": [], "train": [], "val": []}
    t0 = time.time()

    for step in range(CONFIG["max_iters"] + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, CONFIG["lr"], CONFIG["warmup_iters"])

        if step % CONFIG["eval_interval"] == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                CONFIG["block_size"],
                CONFIG["batch_size"],
                CONFIG["eval_iters"],
                device,
            )
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            elapsed = time.time() - t0
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {elapsed:.1f}s"
            )

        x, y = get_batch(
            train_data, CONFIG["block_size"], CONFIG["batch_size"], device
        )
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # 保存
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "mini_gpt_toy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": CONFIG,
            "vocab_size": vocab_size,
            "stoi": stoi,
        },
        ckpt_path,
    )
    print(f"\n模型已保存: {ckpt_path}")

    # 生成样例
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generate(context, max_new_tokens=300, temperature=0.8)[0].tolist()
    sample_text = decode(sample_ids, itos)
    sample_path = RESULTS_DIR / "sample.txt"
    sample_path.write_text(sample_text, encoding="utf-8")
    print(f"\n=== 生成样例 (前 200 字) ===\n{sample_text[:200]}")
    print(f"\n完整样例已写入: {sample_path}")

    # loss 曲线
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(history["step"], history["train"], label="train")
        plt.plot(history["step"], history["val"], label="val")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("MiniGPT on tiny_shakespeare")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=120)
        print(f"loss 曲线已保存: {RESULTS_DIR / 'loss_curve.png'}")
    except ImportError:
        print("(matplotlib 未装,跳过 loss 曲线绘制)")


if __name__ == "__main__":
    main()
