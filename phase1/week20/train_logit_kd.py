"""
Phase1 Week20 Part A: Logit-KD 训练 (transformers.Trainer + 自定义 compute_loss + PEFT-LoRA)

在 CPT baseline (week12_lora_cpt/50_50_fused) 上做 logit-level knowledge distillation.
loss = α·CE(student, gold) + (1−α)·T²·KL_restricted(teacher_topk ‖ student_topk)
  (kd_loss.py). teacher top-K logits 来自 extract_teacher_logits.py (离线 MLX forward).

★ 与 week19 distill 臂同 prompt / 同 completion / 同 base / 同 eval → 唯一变量是 loss
  (hard CE vs soft KL), 干净对照 "学 teacher 分布 vs 学 teacher argmax".

设计:
  - 不用 SFTTrainer (要加 teacher_topk 列 + 自定义 loss, SFT 的 prompt/completion 流程会干扰).
  - 自定义 KDDataset (distill_sft.jsonl + teacher_topk_logits.jsonl 按 question_id 对齐).
  - 自定义 collator: 右 pad, labels(prompt=-100, completion=token_id),
    teacher_topk 填在 seq position L-1..L+M-2 (causal shift 后对齐 completion token).
  - KDTrainer.compute_loss: forward → 3 者 shift → kd_loss.
  - tokenizer 一致性 sanity: student HF encode(prompt) len 必须等于 teacher 的 prompt_len.

脚手架 (offline env / get_device / loss_log / run_config) 镜像 week19 train_distill_sft.py.

Usage:
  # smoke
  phase1/.venv/bin/python phase1/week20/train_logit_kd.py \\
    --model phase1/results/week12_lora_cpt/50_50_fused \\
    --data phase1/results/week19_distill/data/distill_sft.jsonl \\
    --logits phase1/results/week20_distill/data/teacher_topk_logits.jsonl \\
    --output phase1/results/week20_distill/_smoke_kd \\
    --alpha 0.5 --temperature 2.0 --limit 32 --max-steps 5
  # full arm (kd_t2)
  ... --alpha 0.5 --temperature 2.0 --output phase1/results/week20_distill/kd_t2
"""

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

# 国内 HF 网络会 hang; 模型/数据全本地 → 强制 offline (week15/17/19 同款坑)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# MPS 兜底 (torch import 前设): MTL_TIMEOUT=0 禁 Metal 超时; MPS_FALLBACK=1 未实现算子回退 CPU
os.environ.setdefault("MTL_TIMEOUT", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# env vars (上面已设) 必须先于 torch import 生效
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))  # kd_loss 同目录
from kd_loss import kd_loss  # noqa: E402


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MPSEmptyCacheCallback(__import__("transformers").TrainerCallback):
    """MPS caching allocator 在变长 batch 下碎片化 → free pool 单调涨到天花板 OOM
    (week20 实测: 三臂都在 step106 崩, MPS allocated 22GB + other 66GB > 88GB max).
    每 N 步释放一次 free pool — PyTorch 官方推荐的 MPS 防 OOM 手法 (torch.mps.empty_cache),
    只释放未被引用的缓存, 不动活跃 tensor, 安全且开销极小.
    """

    def __init__(self, every: int = 10):
        self.every = max(1, every)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            try:
                torch.mps.empty_cache()
                # 证据: empty_cache 后 pool 该持平不涨 (修复前涨到 88GB OOM)
                alloc = torch.mps.current_allocated_memory() / (1024 ** 3)
                print(f"[mps-mem] step={state.global_step} | empty_cache 后 MPS allocated={alloc:.2f} GB", flush=True)
            except Exception:
                pass


@dataclass
class KDSample:
    question_id: int
    prompt_ids: list          # student HF encode(prompt)
    comp_ids: list            # teacher 存的 completion_token_ids (同 vocab, 已对齐)
    teacher_tokens: list      # [M][K] token ids, teacher top-K per completion position
    teacher_logits: list      # [M][K] raw logits


class KDDataset:
    """distill_sft.jsonl ⊕ teacher_topk_logits.jsonl → KDSample list."""

    def __init__(self, sft_path: str, logits_path: str, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # teacher logits index
        tlogits = {}
        for line in Path(logits_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                tlogits[r["question_id"]] = r
        # sft rows
        sft = [json.loads(l) for l in Path(sft_path).read_text(encoding="utf-8").splitlines() if l.strip()]

        self.samples = []
        n_mismatch = 0
        for r in sft:
            qid = r["question_id"]
            if qid not in tlogits:
                continue
            tr = tlogits[qid]
            prompt_ids = tokenizer.encode(r["prompt"])
            # ★ tokenizer 一致性 sanity: student HF encode(prompt) len == teacher prompt_len
            if len(prompt_ids) != tr["prompt_len"]:
                n_mismatch += 1
                continue
            comp_ids = tr["completion_token_ids"]
            positions = tr["positions"]
            # 截断 keep_start (completion 字母在前, 截尾丢解释不丢字母)
            total = len(prompt_ids) + len(comp_ids)
            if total > max_length:
                keep = max_length - len(prompt_ids)
                if keep <= 0:
                    continue
                comp_ids = comp_ids[:keep]
                positions = positions[:keep]
            assert len(comp_ids) == len(positions), f"qid={qid} comp {len(comp_ids)} != pos {len(positions)}"
            self.samples.append(KDSample(
                question_id=qid,
                prompt_ids=prompt_ids,
                comp_ids=comp_ids,
                teacher_tokens=[p["topk_tokens"] for p in positions],
                teacher_logits=[p["topk_logits"] for p in positions],
            ))
        if n_mismatch:
            print(f"[kd-data] ⚠ {n_mismatch} 条 prompt_len 不匹配 (student/teacher tokenizer 不一致?), 已跳过", flush=True)
        print(f"[kd-data] 加载 {len(self.samples)} 条 (sft {len(sft)} / teacher {len(tlogits)}) | max_length={max_length}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@dataclass
class KDCollator:
    """右 pad; teacher_topk 填在 seq position L-1..L+M-2 (causal shift 后对齐 completion token)."""
    pad_id: int
    k: int

    def __call__(self, samples: list[KDSample]) -> dict:
        B = len(samples)
        max_seq = max(len(s.prompt_ids) + len(s.comp_ids) for s in samples)
        L_max = max_seq
        K = self.k
        input_ids = torch.full((B, L_max), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L_max), dtype=torch.long)
        labels = torch.full((B, L_max), -100, dtype=torch.long)
        teacher_tokens = torch.zeros((B, L_max, K), dtype=torch.long)
        teacher_logits = torch.zeros((B, L_max, K), dtype=torch.float32)

        for b, s in enumerate(samples):
            L = len(s.prompt_ids)
            M = len(s.comp_ids)
            seq = L + M
            input_ids[b, :seq] = torch.tensor(s.prompt_ids + s.comp_ids, dtype=torch.long)
            attn[b, :seq] = 1
            labels[b, L:seq] = torch.tensor(s.comp_ids, dtype=torch.long)
            # teacher top-K: completion token c_t (input pos L+t) 由 logits[pos L+t-1] 预测
            # → 填 teacher position t 到 seq pos L-1+t
            for t in range(M):
                pos = L - 1 + t
                teacher_tokens[b, pos] = torch.tensor(s.teacher_tokens[t], dtype=torch.long)
                teacher_logits[b, pos] = torch.tensor(s.teacher_logits[t], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "teacher_tokens": teacher_tokens,
            "teacher_logits_topk": teacher_logits,
        }


class KDTrainer(__import__("transformers").Trainer):
    """compute_loss: forward → 3 者 causal shift → kd_loss; log ce/kl 分量."""

    def __init__(self, *args, kd_alpha: float = 0.5, kd_temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self._buf_ce = 0.0
        self._buf_kl = 0.0
        self._buf_n = 0
        self._dbg_calls = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch
        input_ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        labels = inputs["labels"]
        t_tok = inputs["teacher_tokens"]
        t_log = inputs["teacher_logits_topk"]

        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits                      # (B, seq, V)

        # causal shift: 3 者对齐 (logits[i] 预测 input[i+1])
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        t_tok = t_tok[:, :-1, :]
        t_log = t_log[:, :-1, :]

        loss, ce, kl = kd_loss(
            logits, labels, t_tok, t_log,
            alpha=self.kd_alpha, temperature=self.kd_temperature,
        )
        # 一次性 debug: 验证 kd_loss total == α·ce + (1-α)·T²·kl, 定位 loss logging 口径
        if os.environ.get("KD_DEBUG") and self._dbg_calls < 4:
            t_total = self.kd_alpha * ce.item() + (1.0 - self.kd_alpha) * (self.kd_temperature ** 2) * kl.item()
            print(f"[kd-dbg] call#{self._dbg_calls} | kd_loss.total={loss.item():.4f} | "
                  f"α·ce+(1-α)·T²·kl={t_total:.4f} | ce={ce.item():.4f} kl={kl.item():.4f} "
                  f"| nitems={num_items_in_batch}", flush=True)
            self._dbg_calls += 1
        self._buf_ce += float(ce.detach())
        self._buf_kl += float(kl.detach())
        self._buf_n += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self._buf_n > 0:
            ce_m = self._buf_ce / self._buf_n
            kl_m = self._buf_kl / self._buf_n
            logs["ce"] = round(ce_m, 4)
            logs["kl"] = round(kl_m, 4)
            # 重算 loss 保证 loss == α·ce + (1-α)·T²·kl (transformers 5.x 在部分累积步
            # logged loss 口径不一致 — smoke 小样本 / full 整除都通用, 报告诚实)
            logs["loss"] = round(
                self.kd_alpha * ce_m + (1.0 - self.kd_alpha) * (self.kd_temperature ** 2) * kl_m, 4
            )
            self._buf_ce = 0.0
            self._buf_kl = 0.0
            self._buf_n = 0
        super().log(logs, start_time)


def parse_args():
    p = argparse.ArgumentParser(description="Logit-KD 训练 (Trainer + PEFT-LoRA)")
    p.add_argument("--model", required=True, help="CPT baseline (50_50_fused)")
    p.add_argument("--data", required=True, help="week19 distill_sft.jsonl")
    p.add_argument("--logits", required=True, help="extract_teacher_logits 产物 teacher_topk_logits.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=0.5, help="CE 权重 (1=纯hard CE week19, 0=纯KL)")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-5, help="与 week19 SFT 同")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="有效 batch=16 (= week19/17)")
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=10, help="smoke 设 1 看逐步 loss")
    p.add_argument("--max-steps", type=int, default=0, help="smoke (>0 覆盖 epochs)")
    p.add_argument("--limit", type=int, default=0, help="只取前 N (smoke)")
    # LoRA (与 DPO/CPT/GRPO/SFT 一致)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = get_device()
    random.seed(args.seed)
    print(f"[kd-train] device={device} | α={args.alpha} | T={args.temperature} | topk={args.topk} | "
          f"lr={args.lr} | epochs={args.epochs} | effective_batch={args.batch_size*args.grad_accum}", flush=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[kd-train] 加载 baseline: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch_dtype).to(device)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 数据
    ds = KDDataset(args.data, args.logits, tokenizer, args.max_length)
    if args.limit and args.limit > 0:
        ds.samples = ds.samples[: args.limit]
        print(f"[kd-train] limit → {len(ds.samples)} 条 (smoke)", flush=True)
    collator = KDCollator(pad_id=tokenizer.pad_token_id, k=args.topk)

    mp_kwargs = {"bf16": True} if args.dtype == "bfloat16" else ({"fp16": True} if args.dtype == "float16" else {})
    targs = TrainingArguments(
        output_dir=args.output,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,           # 省 MPS 显存
        logging_steps=args.logging_steps,
        save_strategy="no",                    # 末尾 trainer.save_model()
        seed=args.seed,
        report_to="none",
        dataloader_pin_memory=False,           # MPS 不支持 (week19 run.log 警告)
        remove_unused_columns=False,           # 自定义 collator, 别删列
        **mp_kwargs,
    )
    if args.max_steps and args.max_steps > 0:
        targs.max_steps = args.max_steps

    trainer = KDTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[MPSEmptyCacheCallback(every=args.logging_steps)],
        kd_alpha=args.alpha,
        kd_temperature=args.temperature,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # loss_log.csv + run_config.json (镜像 train_distill_sft)
    hist = trainer.state.log_history
    log_fields = ["step", "epoch", "loss", "ce", "kl", "grad_norm", "learning_rate"]
    with open(os.path.join(args.output, "loss_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=log_fields, extrasaction="ignore")
        w.writeheader()
        for row in hist:
            w.writerow({k: row.get(k, "") for k in log_fields})

    run_cfg = {
        "method": "logit_distillation_KD",
        "model": args.model, "data": args.data, "logits": args.logits,
        "alpha": args.alpha, "temperature": args.temperature, "topk": args.topk,
        "lr": args.lr, "epochs": args.epochs, "max_steps": args.max_steps, "limit": args.limit,
        "max_length": args.max_length, "dtype": args.dtype,
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "seed": args.seed, "n_samples": len(ds.samples), "device": device,
    }
    Path(args.output, "run_config.json").write_text(
        json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    last = next((r for r in reversed(hist) if "loss" in r), {})
    print(f"\n[kd-train] ✓ 完成 -> {args.output}", flush=True)
    if last:
        print(f"[kd-train] 末步 loss={last.get('loss','-')} ce={last.get('ce','-')} "
              f"kl={last.get('kl','-')} grad_norm={last.get('grad_norm','-')}", flush=True)


if __name__ == "__main__":
    main()
