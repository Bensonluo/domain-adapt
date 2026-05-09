# Phase 0 GPU 实验通用手册

> 适用平台: AutoDL / 腾讯云 / 阿里云 / RunPod / Lambda 等任意 Linux + CUDA 服务器
> 不绑定任何云厂商,命令是通用的。

## 1. 首次上机: 环境初始化

```bash
# 确认 CUDA
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 安装依赖 (如果镜像里没预装 torch,就取消注释第一行)
cd /root/workspace/growing-big   # 你上传代码的路径
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r phase0/requirements-gpu.txt
```

### 常见云镜像差异速查

| 平台 | 镜像选择 | 备注 |
|------|----------|------|
| AutoDL | PyTorch 2.4 / CUDA 12.1 | 社区镜像最丰富 |
| 腾讯云 | GN10X + Ubuntu 22.04 | 需手动装 conda/pip |
| 阿里云 | ecs.gn6 系列 + DeepLearning 镜像 | 预装 torch |
| RunPod | PyTorch 2.x 模板 | 直接可用 |

## 2. 上传/下载数据与代码

### 方式 A: rsync (推荐,Mac/Linux 直接)

```bash
# 上传本地代码到服务器 (在项目根目录执行)
rsync -avz --exclude='.venv/' --exclude='phase0/checkpoints/' --exclude='phase0/data/' \
    /Users/luopeng/Documents/GitHub/growing-big/ \
    root@your-server-ip:/root/workspace/growing-big/

# 下载训练产物回本地
rsync -avz root@your-server-ip:/root/workspace/growing-big/phase0/checkpoints/ \
    /Users/luopeng/Documents/GitHub/growing-big/phase0/checkpoints/
```

### 方式 B: 云厂商自带网盘 (AutoDL 网盘 / 腾讯云 COS)

- AutoDL: 文件传到 `/root/autodl-tmp/` 持久化
- 腾讯云: 用 `coscli` 或 Web 控制台上传
- 阿里云: `ossutil` 或 Web 控制台

### 方式 C: HuggingFace Hub 中转 (数据量小时)

```python
# 把数据集 push 到 HF Hub,云端直接 pull
from datasets import load_dataset
ds = load_dataset("your-username/medical-sft-data")
```

## 3. 训练启动模板

### Week 3: Qwen-1.5B 全量微调

```bash
# 在 GPU 服务器上
cd /root/workspace/growing-big/phase0/week3
python train_full_ft.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data /path/to/domain_data.jsonl \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 2 \
    --lr 5e-5

# 监控: 另一个窗口
watch -n 2 nvidia-smi
```

### Week 5-6: Qwen-3B QLoRA 训练

```bash
cd /root/workspace/growing-big/phase0/week5
python train_qlora.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data /path/to/domain_data.jsonl \
    --output_dir ./qlora \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32

# 合并 adapter -> 完整模型
python merge_adapter.py --adapter ./qlora --output ./merged
```

### Week 8: lm-eval 在训练模型上重跑

```bash
cd /root/workspace/growing-big
python phase0/utils/eval_baseline.py \
    --model ./phase0/week6/merged \
    --output phase0/results/eval_after_sft.json
```

## 4. 显存监控与调参

```bash
# 实时监控
nvidia-smi -l 2

# PyTorch 里获取峰值显存
torch.cuda.max_memory_allocated() / 1024**3  # GB

# 训练脚本里常用
from accelerate import Accelerator
accelerator = Accelerator()
print(f"device: {accelerator.device}, memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

## 5. 断线重连 / 后台训练

```bash
# 用 tmux / screen 防断线
tmux new -s train
python train_qlora.py ...
# Ctrl+B, D  detach
tmux attach -t train
```

## 6. 费用估算

| 实验 | GPU | 预估时长 | 预估费用 |
|------|-----|----------|----------|
| Week 3 全量微调 1.5B | 4090 24GB | 2-4h | ¥5-15 |
| Week 5 QLoRA 3B | 4090 24GB | 3-6h | ¥8-20 |
| Week 6 完整 SFT + eval | 4090 24GB | 5-10h | ¥15-40 |
| Week 8 lm-eval 重跑 | CPU / 小 GPU | 1-2h | ¥3-5 |
| **8 周总计** | — | — | **约 ¥50-150** (AutoDL 价) |

> 腾讯云/阿里云同规格贵 2-3 倍。
