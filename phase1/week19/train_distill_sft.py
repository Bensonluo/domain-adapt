"""
Phase 1 Week 19: Response-Distillation SFT 训练 (TRL SFTTrainer)

在 CPT baseline (week12_lora_cpt/50_50_fused) 上做 instruction SFT,
completion = teacher(35B) 或人写答案 (字母+解释)。三臂对照 (real/distill/mixed)。

核心机制 (TRL v1.8.0 本地源码核验, sft_config.py / sft_trainer.py):
  1. **completion_only_loss**: {prompt,completion} 数据自动开启 (sft_trainer.py:1190-1191),
     loss 只算 completion token (prompt token label=-100)。显式传 True 更稳。
  2. **peft_config= 传 SFTTrainer, 不要自己 get_peft_model** (sft_trainer.py:938, 同 GRPO)。
  3. **max_length=1536** (默认 1024 截 p99 长医学解释); truncation_mode="keep_start" 保字母。
  4. **packing=False** (completion_only_loss 要求, 否则 mask 语义破坏)。
  5. bf16 (MPS torch2.13 已支持), gradient_checkpointing (默认, 省 MPS 显存)。

脚手架镜像 phase1/week17/train_grpo.py (offline env / get_device / loss_log / run_config)。

Usage:
    python phase1/week19/train_distill_sft.py \\
        --model phase1/results/week12_lora_cpt/50_50_fused \\
        --data phase1/results/week19_distill/data/real_sft.jsonl \\
        --output phase1/results/week19_distill/real
"""

import argparse
import csv
import json
import os
from pathlib import Path

# 国内 HF 网络会 hang; 模型/数据全本地 → 强制 offline (week15/17 同款坑)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# MPS 兜底 (torch import 前设): MTL_TIMEOUT=0 禁 Metal 超时; MPS_FALLBACK=1 未实现算子回退 CPU
os.environ.setdefault("MTL_TIMEOUT", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="蒸馏 SFT (TRL SFTTrainer + PEFT-LoRA)")
    p.add_argument("--model", required=True, help="CPT baseline (50_50_fused)")
    p.add_argument("--data", required=True, help="{prompt, completion} jsonl")
    p.add_argument("--output", required=True)
    # SFT 超参 (plan, 含依据)
    p.add_argument("--lr", type=float, default=2e-5,
                   help="TRL SFT 默认; LoRA-SFT 区间 1e-5~3e-5 (比 GRPO 1e-5 高, SFT 梯度更密)")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="有效 batch=16 (= week17 GRPO, 可比)")
    p.add_argument("--max-length", type=int, default=1536,
                   help="默认 1024 截 p99 长医学解释; keep_start 保字母")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=0, help="smoke 用 (>0 覆盖 epochs)")
    p.add_argument("--limit", type=int, default=0, help="只取前 N 条 (smoke)")
    # LoRA (与 DPO/CPT/GRPO 一致)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = get_device()
    print(f"[sft] device={device} | lr={args.lr} | epochs={args.epochs} | "
          f"max_length={args.max_length} | effective_batch={args.batch_size * args.grad_accum}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[sft] 加载 baseline: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ⚠ 不 get_peft_model! SFTTrainer 用 peft_config= 内部 wrap (同 GRPO)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype).to(device)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # 读 {prompt, completion} → 只取这两列 (SFTTrainer 自动判 prompt/completion → completion_only_loss)
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    print(f"[sft] SFT 数据 {len(ds)} 条 (prompt/completion)")

    mp_kwargs = {"bf16": True} if args.dtype == "bfloat16" else ({"fp16": True} if args.dtype == "float16" else {})
    cfg = SFTConfig(
        output_dir=args.output,
        # SFT 核心 (TRL v1.8.0 字段, 已源码核验)
        completion_only_loss=True,          # loss 只在 completion token (prompt mask 掉)
        max_length=args.max_length,         # 提到 1536 (默认 1024 截 p99)
        packing=False,                      # completion_only_loss 要求
        truncation_mode="keep_start",       # 截尾保字母 (completion 字母在前)
        # 优化
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,        # 省 MPS 显存
        # 日志/存盘
        logging_steps=10,
        save_strategy="no",                 # 末尾 trainer.save_model()
        seed=args.seed,
        report_to="none",
        **mp_kwargs,
    )
    if args.max_steps and args.max_steps > 0:
        cfg.max_steps = args.max_steps      # smoke 覆盖 epochs

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,               # ← QLoRA 在这传, 不是 get_peft_model
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # loss_log.csv + run_config.json (镜像 train_grpo)
    hist = trainer.state.log_history
    log_fields = ["step", "epoch", "loss", "grad_norm", "learning_rate"]
    with open(os.path.join(args.output, "loss_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
        w.writeheader()
        for row in hist:
            w.writerow({k: row.get(k, "") for k in log_fields})

    run_cfg = {
        "model": args.model, "data": args.data, "lr": args.lr, "epochs": args.epochs,
        "max_steps": args.max_steps, "limit": args.limit, "dtype": args.dtype,
        "max_length": args.max_length, "completion_only_loss": True, "packing": False,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "n_samples": len(ds), "device": device,
    }
    Path(args.output, "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    last = next((r for r in reversed(hist) if "loss" in r), {})
    print(f"\n[sft] ✓ 完成 -> {args.output}")
    if last:
        print(f"[sft] 末步 loss={last.get('loss', '-')} grad_norm={last.get('grad_norm', '-')}")


if __name__ == "__main__":
    main()
