"""
Phase 1 Week 15: DPO 训练脚本（TRL DPOTrainer）

在 CPT baseline (week12_lora_cpt/50_50_fused) 上做 Direct Preference Optimization。
扫 β ∈ {0.1, 0.3, 0.5}，回答「对齐强度 vs 通用遗忘的 trade-off」。

Usage:
    # smoke (30 对, 验证管线)
    python phase1/week15/train_dpo.py \
        --model phase1/results/week12_lora_cpt/50_50_fused \
        --data phase1/data/processed/preference/train_split.jsonl \
        --beta 0.3 --limit 30 --output phase1/results/week15_dpo_smoke

    # 正式 (全量 1299 对, 1 epoch)
    python phase1/week15/train_dpo.py \
        --model phase1/results/week12_lora_cpt/50_50_fused \
        --beta 0.3 --output phase1/results/week15_dpo/beta_0.3

设计要点 (全部已查证, 见 plan + week14/trl_source_notes.md):
  1. **bf16 不是 bf16→fp16**: torch 2.13 MPS 已支持 bf16 (实测). bf16 指数域同 fp32,
     无 underflow、无需 GradScaler, 是 Apple Silicon DPO 的最优 dtype.
  2. **PEFT LoRA + ref_model=None**: DPOTrainer 检测到 is_peft_model 且 ref_model=None
     时, 用 adapter-disable 把基模当 reference (源码笔记 §1.3), 0 额外显存.
  3. **lr=5e-6**: LoRA DPO 比 full-param (5e-7) 高一个量级 (adapter 随机初始化可吃更大 lr).
  4. reward margin/accuracies 由 DPOTrainer 自动 log, 训完从 log_history 落盘.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

# 国内 HF 网络会 hang (transformers/trl 启动会 phone home)。模型/tokenizer/数据全本地，
# 强制 offline（week12 同款坑，已实测：不开 offline 训练步卡死在 9% CPU）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def get_device() -> str:
    """检测可用的计算设备。"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="DPO 训练 (TRL + PEFT-LoRA)")
    p.add_argument("--model", required=True, help="CPT baseline 模型路径 (HF 格式)")
    p.add_argument("--data", default="phase1/data/processed/preference/train_split.jsonl",
                   help="偏好数据 jsonl ({prompt,chosen,rejected})")
    p.add_argument("--beta", type=float, default=0.3, help="DPO beta (对齐强度)")
    p.add_argument("--lr", type=float, default=5e-7, help="Learning rate (LoRA 建议 5e-6)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="只取前 N 对 (smoke 用, 0=全部)")
    p.add_argument("--noise", type=float, default=0.0,
                   help="翻转此比例的偏好标签 (week16 失败模式钩子)")
    p.add_argument("--output", required=True, help="输出目录")
    # LoRA (与 CPT 对齐: rank16/alpha32/dropout0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # DPO / Trainer 超参
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=4096,
                   help="TRL 1.x 用 max_length + truncation_mode=keep_start (无独立 prompt cap)")
    p.add_argument("--loss-type", type=str, default="sigmoid",
                   choices=["sigmoid", "ipo", "hinge", "robust"], help="sigmoid=DPO, ipo=长度归一(week16)")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="bfloat16 最优(MPS 已支持); float32 兜底")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def inject_noise(rows, frac, seed):
    """翻转 frac 比例的 chosen/rejected, 返回 (新 rows, 被翻转的 index 列表)。"""
    if frac <= 0:
        return rows, []
    n = len(rows)
    k = int(round(n * frac))
    rng = random.Random(seed)
    flip_idx = rng.sample(range(n), k)
    for i in flip_idx:
        rows[i]["chosen"], rows[i]["rejected"] = rows[i]["rejected"], rows[i]["chosen"]
    return rows, flip_idx


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = get_device()
    print(f"[dpo] device={device} | beta={args.beta} | lr={args.lr} | dtype={args.dtype} "
          f"| loss={args.loss_type} | noise={args.noise}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # 1. 加载 model + tokenizer (CPT baseline)
    print(f"[dpo] 加载 baseline: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype).to(device)

    # 2. PEFT LoRA → ref_model=None (adapter-disable 当 reference, 0 额外显存)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3. 加载偏好数据 (可选 limit / noise)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if args.noise > 0:
        rows, flipped = inject_noise(rows, args.noise, args.seed)
        print(f"[dpo] ⚠ 注入 {len(flipped)}/{len(rows)} ({args.noise*100:.0f}%) 噪声 (翻转 chosen/rejected)")
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
    print(f"[dpo] 偏好数据 {len(ds)} 对")

    # 4. DPOConfig
    mp_kwargs = {}
    if args.dtype == "bfloat16":
        mp_kwargs["bf16"] = True
    elif args.dtype == "float16":
        mp_kwargs["fp16"] = True

    cfg = DPOConfig(
        output_dir=args.output,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        loss_type=args.loss_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        **mp_kwargs,
    )

    # 5. DPOTrainer + 训练
    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # PEFT → adapter-disable 当 reference
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # 6. 落盘: loss_log.csv (含 reward margin/accuracies) + run_config.json
    hist = trainer.state.log_history
    log_fields = ["step", "epoch", "loss", "rewards/chosen", "rewards/rejected",
                  "rewards/margins", "rewards/accuracies", "learning_rate"]
    with open(os.path.join(args.output, "loss_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
        w.writeheader()
        for row in hist:
            w.writerow({k: row.get(k, "") for k in log_fields})

    run_cfg = {
        "model": args.model, "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
        "limit": args.limit, "noise": args.noise, "dtype": args.dtype, "loss_type": args.loss_type,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "seed": args.seed, "n_pairs": len(ds), "device": device,
    }
    Path(args.output, "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 末步 reward 统计 (快速 sanity)
    last = next((r for r in reversed(hist) if "rewards/margins" in r), {})
    print(f"\n[dpo] ✓ 完成 -> {args.output}")
    if last:
        print(f"[dpo] 末步 reward margin={last.get('rewards/margins', '-'):.4f} "
              f"accuracy={last.get('rewards/accuracies', '-'):.3f}")


if __name__ == "__main__":
    main()
