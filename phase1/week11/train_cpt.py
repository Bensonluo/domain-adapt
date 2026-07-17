"""
Phase 1 Week 11: CPT 训练脚本 (Apple MLX 路线)
=============================================

在 Apple Silicon (M3 Max) 上用 MLX 框架做 Continual Pre-training。

模型: Qwen/Qwen3.5-0.8B (base, 全量 CPT 用非量化原版)
框架: mlx-lm (Apple 官方, 适配 Apple Silicon, 比 transformers+torch 在 Mac 上快很多)
数据: week10 产出的 cpt_{ratio}.jsonl (text 字段, 自动转 MLX 的 train.jsonl/valid.jsonl)

核心学习目标:
1. 理解 CPT 和 SFT 在数据格式上的区别 (CPT 用纯 text, 不套 chat template)
2. 理解为什么 CPT 学习率比预训练小 10-100 倍 (1e-5)
3. 观察 CPT 的 loss 曲线 (全量 vs LoRA 的不同)

实时 loss 曲线 (三种方式, 任选):
  默认              实时 stdout 回显 + loss_log.csv + 训练后 loss_curve.png
  --live-plot       matplotlib 实时窗口 (本地零登录, 窗口实时刷新曲线)
  --report-to swanlab  SwanLab web dashboard (国内友好, 最专业的实时曲线)

用法:
    # 基础全量 CPT (默认 70-30 配比, 200 iters 验证流程):
    python phase1/week11/train_cpt.py

    # 实时 matplotlib 窗口:
    python phase1/week11/train_cpt.py --live-plot

    # SwanLab web 曲线 (需 pip install swanlab):
    python phase1/week11/train_cpt.py --report-to swanlab

    # LoRA-CPT (省内存, 改 fine-tune-type):
    python phase1/week11/train_cpt.py --fine-tune-type lora

环境 (训练前一次性准备):
    pip install "mlx-lm[train]"          # MLX 训练框架
    pip install matplotlib                # 可选: --live-plot / loss_curve.png
    pip install swanlab                   # 可选: --report-to swanlab
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "phase1" / "data" / "processed" / "cpt_ready"   # week10 输出
DEFAULT_MLX_DATA = ROOT / "phase1" / "data" / "processed" / "cpt_mlx"  # MLX 消费目录
DEFAULT_OUTPUT = ROOT / "phase1" / "results" / "week11_cpt_pure"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"   # 纯 base (非 Instruct); 全量 CPT 必须非量化原版


# ─────────────────────────────────────────────
# 1. 数据准备: week10 cpt_{ratio}.jsonl → MLX train.jsonl + valid.jsonl
# ─────────────────────────────────────────────


def _write_text_jsonl(path: Path, texts: list[str]) -> None:
    """写 MLX text 格式 jsonl, 每行 {"text": "..."}。多余字段 (ids/n_tokens) 不写。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def prepare_mlx_data(
    ratio: str, src_dir: Path, mlx_dir: Path, val_ratio: float = 0.1, min_val: int = 4
) -> tuple[int, int]:
    """把 week10 的 cpt_{ratio}.jsonl 转成 MLX 要求的 train.jsonl + valid.jsonl。

    MLX 要求数据目录下有 train.jsonl (必须) + valid.jsonl (可选, 用于训练中 val loss)。
    text 格式: CPT 不 mask prompt, 每个token都算 loss (区别于 SFT 的 completion-only)。

    min_val: 验证集最少条数 (默认 4)。MLX evaluate 要凑齐一个 batch_size 才不报错
    ("Dataset must have at least batch_size examples")。demo 数据小 (16 条),
    10% 切出来可能 < batch_size, 所以保底至少 min_val 条, 但不超过总数一半。
    """
    src = src_dir / f"cpt_{ratio}.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"找不到 {src}。先跑 week10/data_prep_cpt.py 生成 CPT 数据。\n"
            f"  python phase1/week10/data_prep_cpt.py"
        )

    records = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [r["text"] for r in records]   # 只取 text, 丢 ids/n_tokens

    # 切验证集: 至少 min_val 条 (凑齐 batch_size), 但不超过总数一半 (留给训练)
    n_val = min(max(min_val, int(len(texts) * val_ratio)), len(texts) // 2)
    n_val = max(1, n_val)   # 兜底: 极端小数据至少 1 条
    val_texts, train_texts = texts[:n_val], texts[n_val:]

    _write_text_jsonl(mlx_dir / "train.jsonl", train_texts)
    _write_text_jsonl(mlx_dir / "valid.jsonl", val_texts)
    return len(train_texts), len(val_texts)


# ─────────────────────────────────────────────
# 2. 构造 mlx_lm.lora 训练命令 (CLI flags, 字段名来自 mlx_lm/lora.py 源码)
# ─────────────────────────────────────────────


def build_train_command(args, mlx_data_dir: Path, adapter_path: Path) -> list[str]:
    """构造 `python -m mlx_lm lora ...` 命令。

    关键: 用 `python -m mlx_lm lora` (空格), 不是 `python -m mlx_lm.lora` (已 deprecated)。
    全量 CPT = --fine-tune-type full --num-layers -1 (解冻所有层)。
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", str(mlx_data_dir),
        "--fine-tune-type", args.fine_tune_type,
        "--num-layers", str(args.num_layers),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
        "--max-seq-length", str(args.max_seq_length),
        "--steps-per-report", str(args.steps_per_report),
        "--steps-per-eval", str(args.steps_per_eval),
        "--adapter-path", str(adapter_path),
        "--seed", str(args.seed),
    ]
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")          # action="store_true", 开就加 flag
    if args.grad_accum > 1:
        cmd += ["--grad-accumulation-steps", str(args.grad_accum)]
    if args.report_to:
        cmd += ["--report-to", args.report_to]
    if args.config:
        cmd += ["--config", args.config]   # LoRA: lora_parameters (rank/scale/dropout) 只能走 config
    return cmd


# ─────────────────────────────────────────────
# 3. 实时 loss 追踪: 解析 stdout → csv + matplotlib 实时窗口
# ─────────────────────────────────────────────


# mlx-lm 典型输出: "Iter 10: Train loss 2.345, It/sec 5.67" / "Iter 50: Val loss 2.100, Val took 1.2s"
# \w* 兜住 Train/Training, Val/Valid/Validation 等变体
_TRAIN_RE = re.compile(r"(?:Iter|Step)\s*(\d+).*?[Tt]rain\w*\s*loss\s*([\d.eE+-]+)")
_VAL_RE = re.compile(r"(?:Iter|Step)\s*(\d+).*?[Vv]al\w*\s*loss\s*([\d.eE+-]+)")


class LiveLossTracker:
    """实时解析训练 stdout, 写 loss_log.csv + 可选 matplotlib 实时窗口。

    csv 列: iter, kind, loss (kind ∈ train/val)。matplotlib 按 kind 分两条线实时刷新。
    matplotlib 缺失或无 GUI 时自动降级为 csv-only (不报错)。
    """

    def __init__(self, csv_path: Path, live_plot: bool):
        self.csv_path = csv_path
        self.live_plot = live_plot
        self.train_iters, self.train_losses = [], []
        self.val_iters, self.val_losses = [], []
        self.plt = None
        self.fig = None
        self.ax = None
        self._line_train = None
        self._line_val = None

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["iter", "kind", "loss"])

        if live_plot:
            self._init_plot()

    def _init_plot(self):
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            self.plt = plt
            self.fig, self.ax = plt.subplots(figsize=(8, 5))
            self.ax.set_xlabel("iter")
            self.ax.set_ylabel("loss")
            self.ax.set_title("CPT loss (live)")
            self.ax.grid(True, alpha=0.3)
            (self._line_train,) = self.ax.plot([], [], "b-", label="train loss")
            (self._line_val,) = self.ax.plot([], [], "r-", label="val loss")
            self.ax.legend(loc="upper right")
        except ImportError:
            print("⚠️  matplotlib 未安装 → 跳过实时窗口 (pip install matplotlib 启用 --live-plot)")
            self.live_plot = False
        except Exception as e:
            print(f"⚠️  matplotlib 实时窗口初始化失败 ({e}) → 降级为 csv-only")
            self.live_plot = False

    def feed(self, line: str) -> None:
        m = _TRAIN_RE.search(line)
        if m:
            self._record(int(m.group(1)), float(m.group(2)), "train")
            return
        m = _VAL_RE.search(line)
        if m:
            self._record(int(m.group(1)), float(m.group(2)), "val")

    def _record(self, step: int, loss: float, kind: str) -> None:
        self._writer.writerow([step, kind, loss])
        self._csv_file.flush()       # 实时落盘, 用户可随时 tail -f / 导入看曲线

        iters, losses = (self.train_iters, self.train_losses) if kind == "train" else (self.val_iters, self.val_losses)
        iters.append(step)
        losses.append(loss)

        if self.live_plot and self.plt is not None:
            self._line_train.set_data(self.train_iters, self.train_losses)
            self._line_val.set_data(self.val_iters, self.val_losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def finalize(self, png_path: Path) -> None:
        self._csv_file.close()
        # 训练后画最终静态图 (有 matplotlib 才画)
        if not self.train_iters:
            print("⚠️  未解析到任何 loss, 跳过画图 (检查 mlx-lm 输出格式)")
            return
        try:
            import matplotlib
            if self.fig is None:
                matplotlib.use("Agg")   # 无 live 窗口时用无 GUI 后端画静态 png
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.train_iters, self.train_losses, "b-", label="train loss")
            if self.val_iters:
                ax.plot(self.val_iters, self.val_losses, "r-", label="val loss")
            ax.set_xlabel("iter")
            ax.set_ylabel("loss")
            ax.set_title(f"CPT loss ({self.csv_path.parent.name})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
            print(f"✓ loss 曲线: {png_path}")
            if self.live_plot and self.fig is not None:
                self.plt.close(self.fig)
        except ImportError:
            print("⚠️  matplotlib 未安装 → 只写了 csv, 没画 png (pip install matplotlib 启用)")


# ─────────────────────────────────────────────
# 4. 训练 + 生成测试
# ─────────────────────────────────────────────


def run_training(cmd: list[str], output_dir: Path, live_plot: bool) -> int:
    """subprocess 跑 mlx_lm.lora, 实时回显 stdout + 解析 loss。"""
    print("\n" + "=" * 60)
    print("训练命令:")
    print(" ".join(cmd))
    print("=" * 60)

    tracker = LiveLossTracker(output_dir / "loss_log.csv", live_plot)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,   # 行缓冲, 实时拿输出
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")       # 实时回显到终端
            tracker.feed(line)        # 解析 loss
    finally:
        proc.wait()
    tracker.finalize(output_dir / "loss_curve.png")
    return proc.returncode


def run_generation_test(model: str, adapter_path: Path, output_dir: Path) -> None:
    """训练后用 mlx_lm.generate 跑生成测试, 观察领域语言能力变化。"""
    if not adapter_path.exists():
        print("⚠️  adapter 不存在, 跳过生成测试")
        return
    prompts = [
        "患者男性, 54 岁, 主诉胸痛",
        "社区获得性肺炎的常见病原体",
    ]
    out_file = output_dir / "generation_test.txt"
    with out_file.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(f"\n{'='*50}\nPrompt: {p}\n{'='*50}\n")
            cmd = [
                sys.executable, "-m", "mlx_lm", "generate",
                "--model", model,
                "--adapter-path", str(adapter_path),
                "--prompt", p,
                "--max-tokens", "100",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                f.write(result.stdout)
            except subprocess.CalledProcessError as e:
                f.write(f"[生成失败] {e.stderr}\n")
    print(f"✓ 生成测试: {out_file}")


def save_run_config(output_dir: Path, args, mlx_train_n: int, mlx_val_n: int) -> None:
    """保存本次运行配置 (可复现)。"""
    cfg = {
        "model": args.model,
        "fine_tune_type": args.fine_tune_type,
        "num_layers": args.num_layers,
        "ratio": args.ratio,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": args.grad_checkpoint,
        "report_to": args.report_to,
        "mlx_train_samples": mlx_train_n,
        "mlx_val_samples": mlx_val_n,
        "framework": "mlx-lm",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CPT 训练 (Apple MLX 路线)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX/HF 模型 (全量 CPT 用非量化原版)")
    parser.add_argument("--ratio", default="70-30", help="week10 的混合配比 (70-30/50-50/100-0/...)")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="week10 输出目录 (含 cpt_*.jsonl)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出目录")
    # 训练超参 (默认值适配 0.8B + M3 Max 流程验证; 真实训练调大 iters)
    parser.add_argument("--fine-tune-type", default="full", choices=["full", "lora", "dora"],
                        help="full=全量CPT; lora=LoRA-CPT(省内存); 默认 full")
    parser.add_argument("--num-layers", type=int, default=-1,
                        help="解冻层数: -1=全部(全量CPT用); full 模式下默认16只解冻最后16层")
    parser.add_argument("--iters", type=int, default=200, help="训练迭代数 (demo 200; 真实训练 1万+)")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率 (CPT 比预训练小 10-100 倍)")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size (内存不足降到 2/1)")
    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数 (等效大 batch 不增内存)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="序列长度 (匹配 week10 chunk)")
    parser.add_argument("--grad-checkpoint", action="store_true", default=True, help="梯度检查点 (省内存)")
    parser.add_argument("--steps-per-report", type=int, default=10, help="每 N 步打印 loss (曲线数据源)")
    parser.add_argument("--steps-per-eval", type=int, default=50, help="每 N 步算 val loss")
    parser.add_argument("--seed", type=int, default=42)
    # 实时 loss 曲线
    parser.add_argument("--live-plot", action="store_true", help="matplotlib 实时窗口 (本地零登录)")
    parser.add_argument("--report-to", default=None, help="实验追踪 (swanlab 国内友好 / wandb)")
    parser.add_argument("--no-generate", action="store_true", help="跳过训练后生成测试")
    parser.add_argument("--config", default=None,
                        help="mlx_lm YAML 配置路径 (LoRA 的 lora_parameters 只能走 config; CLI flag 优先级更高)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据准备: week10 → MLX train/valid.jsonl
    mlx_data_dir = DEFAULT_MLX_DATA / args.ratio.replace("-", "_")
    n_train, n_val = prepare_mlx_data(
        args.ratio, Path(args.data), mlx_data_dir, min_val=args.batch_size
    )
    print(f"[data] ratio={args.ratio} → train.jsonl ({n_train}) + valid.jsonl ({n_val}) @ {mlx_data_dir}")

    # 2. 检查 mlx-lm 是否装了
    check = subprocess.run([sys.executable, "-c", "import mlx_lm"], capture_output=True)
    if check.returncode != 0:
        print("\n❌ mlx-lm 未安装。先装: pip install \"mlx-lm[train]\"")
        sys.exit(1)

    # 3. 训练
    adapter_path = output_dir / "adapters"
    cmd = build_train_command(args, mlx_data_dir, adapter_path)
    save_run_config(output_dir, args, n_train, n_val)

    rc = run_training(cmd, output_dir, args.live_plot)
    if rc != 0:
        print(f"\n❌ 训练失败 (returncode={rc})。检查命令和日志。")
        sys.exit(rc)

    # 4. 生成测试
    if not args.no_generate:
        run_generation_test(args.model, adapter_path, output_dir)

    print(f"\n✓ 全部完成。结果在 {output_dir}/")
    print(f"  - loss_log.csv / loss_curve.png  (loss 曲线)")
    print(f"  - adapters/                       (训练权重)")
    print(f"  - generation_test.txt             (生成测试)")
    print(f"  - run_config.json                 (可复现配置)")


if __name__ == "__main__":
    main()
