"""
Phase 1 Week 17: GRPO 训练脚本 (TRL GRPOTrainer)

在 CPT baseline (week12_lora_cpt/50_50_fused) 上做 Group Relative Policy Optimization,
reward = MCQ 答对率 (见 reward_functions.py)。产出 GRPO vs DPO (week15/16) 对比。

核心机制 (全部查证 TRL v1.8.0 本地源码, 见 week14/trl_source_notes.md):
  1. **beta=0 不加载 reference model** (grpo_trainer.py L888-894): GRPO 的锚是 *group baseline*
     (advantage = R − mean_group, L2394-2418), 不是 KL 到 ref。beta=0 省 3.4GB 显存。
     [week16 衔接]: β=0.01 在 DPO 上漂移 434 → GRPO 若开 KL (beta>0) 同样需谨慎;
     先跑 beta=0 纯 group-baseline baseline, 漂移过大再加 KL (天然 ablation)。
  2. **loss_type=dapo** (默认): 自带长度偏差消除 (Dynamic Sampling + 无长度奖励), 呼应 week16 IPO。
  3. **QLoRA 传 peft_config= 给 Trainer, 不要自己 get_peft_model** (L411-416 会报错) —— 与 DPO 不同。
  4. **remove_unused_columns=False**: 保留 dataset "answer" 列 → 按列名进 reward kwargs。
  5. **on-policy generation**: 每步对每 prompt 生成 G=4 个 completion, 调 reward → 这是触发 MPS
     F.linear corruption 的路径 (见 plan ⚠️ MPS 风险)。先 smoke (max_steps=2) 验证。

Usage:
    # smoke (风闸, ~10min): 验证 TRL+MPS+generation 不崩, reward 非 NaN
    python phase1/week17/train_grpo.py \\
        --model phase1/results/week12_lora_cpt/50_50_fused \\
        --data phase1/data/processed/cmexam/train.jsonl \\
        --max-steps 2 --limit 4 --output phase1/results/week17_grpo_smoke

    # 正式 (CMExam ~8K, 1 epoch)
    python phase1/week17/train_grpo.py \\
        --model phase1/results/week12_lora_cpt/50_50_fused \\
        --output phase1/results/week17_grpo/mcq_base
"""

import argparse
import csv
import json
import os
from pathlib import Path

# 国内 HF 网络会 hang (transformers/trl 启动 phone home); 模型/数据全本地 → 强制 offline
# (week15 同款坑, 已实测: 不开 offline 训练步卡死)。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# MPS 兜底 (必须在 torch import 前设):
#   MTL_TIMEOUT=0        —— 禁用 Metal GPU 超时 (长 generation 会被系统杀, week11 CPT 同款坑)
#   PYTORCH_ENABLE_MPS_FALLBACK=1 —— 未实现算子回退 CPU (缓解 TRL #4692 LLVM crash; 不救 #180776 静默坏值,
#                          那个靠 smoke 看 reward 是否 NaN/全同来抓)
os.environ.setdefault("MTL_TIMEOUT", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def get_device() -> str:
    """检测可用的计算设备。"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="GRPO 训练 (TRL + PEFT-LoRA, MCQ reward)")
    p.add_argument("--model", required=True, help="CPT baseline 模型路径 (HF 格式)")
    p.add_argument("--data", default="phase1/data/processed/cmexam/grpo_train.jsonl",
                   help="GRPO 数据 jsonl ({prompt, answer}, prepare_cmexam.py 产物)")
    # reward
    p.add_argument("--reward", default="mcq_accuracy",
                   choices=["mcq_accuracy", "format"],
                   help="mcq_accuracy=主线(客观答对率); format=备用(ablation)")
    # GRPO 核心超参
    p.add_argument("--num-generations", type=int, default=4, help="每 prompt 生成数 G (越大显存越大)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-completion-length", type=int, default=48,
                   help="completion 上限。smoke 实测: base(CPT) 在 RAW 格式下首字符即答案字母, "
                        "其后 ramble 编新题不终止 (clipped_ratio=1)。答案=首 token → 48 足够捕获, "
                        "比 256 快 ~5× 且 reward 不变 (dapo 已长度归一)")
    p.add_argument("--kl-coeff", type=float, default=0.0,
                   help="TRL GRPO 的 KL 系数 = beta。beta=0 不加载 ref (默认, 省 3.4GB)")
    p.add_argument("--loss-type", type=str, default="dapo",
                   choices=["dapo", "grpo", "dr_grpo", "bnpo"],
                   help="dapo=默认(自带长度偏差消除); grpo=原版; dr_grpo/bnpo=其他长度归一变体")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="LoRA GRPO lr。实测 (50步 learncheck): lr=1e-6 时 grad_norm~1.2 健康, "
                        "但 reward 零趋势 (LoRA 漂移 ~5e-5 ≈ adapter 幅度 0.1%, 太弱)。"
                        "LoRA-GRPO 文档范围 1e-5~2e-5 (LoRA 需比 full-FT 高 lr, 同 DPO 用 5e-6 之理)")
    # 训练规模
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="只取前 N 条 (smoke 用, 0=全部)")
    p.add_argument("--max-steps", type=int, default=0, help=">0 覆盖 epochs (smoke 用 max_steps=2)")
    # LoRA (与 DPO/CPT 对齐: rank16/alpha32/dropout0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Trainer 超参
    p.add_argument("--batch-size", type=int, default=1, help="per_device_train_batch_size (GRPO 显存大, 建议 1)")
    p.add_argument("--grad-accum", type=int, default=4, help="gradient_accumulation (有效 batch = B×G×accum)")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="bfloat16 最优(MPS 已支持); float32 兜底")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", required=True, help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = get_device()
    print(f"[grpo] device={device} | G={args.num_generations} | kl_coeff(beta)={args.kl_coeff} "
          f"| loss={args.loss_type} | lr={args.lr} | reward={args.reward}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # 本地 reward functions (主线 mcq_accuracy)
    sys_path = str(Path(__file__).resolve().parent)
    import sys
    sys.path.insert(0, sys_path)
    from reward_functions import REWARD_FUNCTIONS

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # 1. 加载 model + tokenizer (CPT baseline)
    print(f"[grpo] 加载 baseline: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ⚠ 不要 get_peft_model! GRPOTrainer 用 peft_config= 内部 wrap (L411-416), 预 wrap 会报错。
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype).to(device)

    # 2. reward funcs (单个 callable; GRPOTrainer 也接 list[Callable], 这里主线用单个)
    reward_fn = REWARD_FUNCTIONS[args.reward]
    print(f"[grpo] reward = {args.reward} ({reward_fn.__name__})")

    # 3. LoRA config (传给 Trainer, 不自己 wrap)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # 4. 加载 GRPO 数据 ({prompt, answer})
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    ds = Dataset.from_list([{"prompt": r["prompt"], "answer": r["answer"]} for r in rows])
    print(f"[grpo] GRPO 数据 {len(ds)} 条 prompt (每条生成 {args.num_generations} 个 completion)")

    # 5. GRPOConfig (v1.8.0)
    mp_kwargs = {}
    if args.dtype == "bfloat16":
        mp_kwargs["bf16"] = True
    elif args.dtype == "float16":
        mp_kwargs["fp16"] = True

    cfg = GRPOConfig(
        output_dir=args.output,
        # GRPO 核心
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        beta=args.kl_coeff,                 # beta=0 → 不加载 reference model (省 3.4GB)
        loss_type=args.loss_type,           # dapo: 自带长度偏差消除
        scale_rewards="group",              # advantage = (R−mean)/std, per-group 归一
        epsilon=0.2,                        # PPO clip
        # 关键: 保留 answer 列进 reward kwargs
        remove_unused_columns=False,
        # 优化
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        # 日志/存盘
        logging_steps=5,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        **mp_kwargs,
    )
    if args.max_steps and args.max_steps > 0:
        cfg.max_steps = args.max_steps          # smoke: 覆盖 epochs
    else:
        cfg.num_train_epochs = args.epochs

    # 6. GRPOTrainer + 训练
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,            # ← QLoRA 在这里传, 不是 get_peft_model
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # 7. 落盘: loss_log.csv (reward 曲线是核心监控) + run_config.json
    hist = trainer.state.log_history
    # GRPO log 字段 (v1.8.0): reward / reward_std / loss / kl / completions_* / learning_rate
    log_fields = ["step", "epoch", "loss", "reward", "reward_std", "kl",
                  "clip_ratio", "learning_rate"]
    with open(os.path.join(args.output, "loss_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
        w.writeheader()
        for row in hist:
            w.writerow({k: row.get(k, "") for k in log_fields})

    run_cfg = {
        "model": args.model, "reward": args.reward, "loss_type": args.loss_type,
        "kl_coeff(beta)": args.kl_coeff, "lr": args.lr, "epochs": args.epochs,
        "max_steps": args.max_steps, "limit": args.limit, "dtype": args.dtype,
        "num_generations": args.num_generations, "temperature": args.temperature,
        "max_completion_length": args.max_completion_length,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "n_prompts": len(ds), "device": device,
    }
    Path(args.output, "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 末步 reward sanity (reward 应非 NaN; smoke 看 reward 是否合理, 非全 0/全 1)
    last_reward = next((r for r in reversed(hist) if "reward" in r), {})
    print(f"\n[grpo] ✓ 完成 -> {args.output}")
    if last_reward:
        print(f"[grpo] 末步 mean reward={last_reward.get('reward', '-')} "
              f"reward_std={last_reward.get('reward_std', '-')} loss={last_reward.get('loss', '-')}")


if __name__ == "__main__":
    main()
