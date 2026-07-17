"""
Phase 1 Week 12: 评估核心 (lm_eval.simple_evaluate + MLXLM + cmmlu 本地化)
==========================================================================

为什么不用 `mlx_lm evaluate` CLI:
  - CLI 没有 --adapter-path (源码 mlx_lm/evaluate.py 的 argparse 确认) → 全量微调产物
    必须先 mlx_lm.fuse 成独立模型
  - CLI 不支持注入 monkeypatch / --include_path → 没法把 cmmlu 重定向到本地数据
  - **hub 1.13.0 在国内连不上**: curl 能下 hf-mirror (resolve 端点可用), 但 Python
    requests 网络栈连不通 (README 小文件都 LocalEntryNotFoundError)。lm-eval 走
    datasets→hub 下数据 → 全挂

解法 (本地化路线): 直接调 lm_eval.simple_evaluate + MLXLM (复用 mlx_lm/evaluate.py
的 model wrapper), monkeypatch datasets.load_dataset 把 'haonan-li/cmmlu' 重定向到
本地解压的 csv (数据已 curl hf-mirror 下来, 存 phase1/data/cmmlu_local/)。

任务选型 (中文医疗 CPT, 数据是中文医学百科):
  - medical_cn (主信号): CMMLU 医学子集, CPT 后应升
  - general_cn (遗忘): CMMLU 非医学子集, CPT 后不应大降
  - medical_en (英文 MMLU 医学, 跨语言检查): 同样依赖 hub, 本周 hub 不通暂跳过

0.8B 噪声: CMMLU 4 选 1 随机基线 25%, 绝对分意义有限 → 看 delta vs base + 固定 seed。
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMMLU_LOCAL = ROOT / "phase1" / "data" / "cmmlu_local"     # 本地解压的 test/dev csv

# ─────────────────────────────────────────────
# 任务组 (加/改只改这里, 可切换/可扩展)
# ─────────────────────────────────────────────
TASK_GROUPS = {
    # 领域信号: 中文医疗 (CPT 后应升)
    "medical_cn": [
        "cmmlu_anatomy",
        "cmmlu_clinical_knowledge",
        "cmmlu_college_medicine",
        "cmmlu_professional_medicine",
        "cmmlu_genetics",                       # CMMLU 叫 genetics, 不是 medical_genetics
        "cmmlu_traditional_chinese_medicine",
        "cmmlu_virology",
        "cmmlu_nutrition",
    ],
    # 遗忘信号: 中文非医学通用 (CPT 后不应大降)
    # 任务名经 lm-eval task_index 核对 (cmmlu 无 econometrics/computer_network):
    #   econometrics → economics; computer_network → marxist_theory (纯文科, 避免科学边界污染遗忘信号)
    "general_cn": [
        "cmmlu_world_history",
        "cmmlu_high_school_physics",
        "cmmlu_economics",
        "cmmlu_marxist_theory",
    ],
    # 跨语言检查: 英文医学 — 同样依赖 hub, 本周 hub 不通暂跳过 (留注释, 不默认跑)
    "medical_en": [
        "mmlu_anatomy",
        "mmlu_clinical_knowledge",
        "mmlu_college_medicine",
        "mmlu_medical_genetics",
        "mmlu_professional_medicine",
    ],
}


def resolve_tasks(groups: list[str]) -> list[str]:
    """把任务组名展开成具体任务列表 (未知组名当裸任务名透传)。去重保序。"""
    tasks: list[str] = []
    for g in groups:
        if g in TASK_GROUPS:
            tasks.extend(TASK_GROUPS[g])
        else:
            tasks.append(g)
    seen, out = set(), []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ─────────────────────────────────────────────
# monkeypatch: cmmlu → 本地 csv (hub 连不上, 数据已 curl 到 CMMLU_LOCAL)
# ─────────────────────────────────────────────
def _install_cmmlu_local_patch() -> None:
    """把 datasets.load_dataset 的 'haonan-li/cmmlu' 请求重定向到本地 csv。

    cmmlu task 调 load_dataset('haonan-li/cmmlu', name=<subject>); 拦截后改走
    load_dataset('csv', data_files={test/dev: <subject>.csv})。csv 列 Question/A/B/C/D/Answer
    与 cmmlu task 的 doc_to_text 模板完全匹配。
    """
    import datasets

    if getattr(datasets.load_dataset, "_cmmlu_patched", False):
        return
    _orig = datasets.load_dataset

    def _patched(path, name=None, **kw):
        if path == "haonan-li/cmmlu" and name:
            if not CMMLU_LOCAL.exists():
                raise FileNotFoundError(
                    f"本地 cmmlu 数据不存在 {CMMLU_LOCAL}. 先跑 download: "
                    "curl -L https://hf-mirror.com/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip"
                )
            return _orig(
                "csv",
                data_files={
                    "test": str(CMMLU_LOCAL / "test" / f"{name}.csv"),
                    "dev": str(CMMLU_LOCAL / "dev" / f"{name}.csv"),
                },
            )
        return _orig(path, name, **kw)

    _patched._cmmlu_patched = True
    datasets.load_dataset = _patched


# ─────────────────────────────────────────────
# 跑评估 (Python API: MLXLM + lm_eval.simple_evaluate)
# ─────────────────────────────────────────────
def run_mlx_evaluate(
    model: str,
    tasks: list[str],
    output_dir: Path,
    limit: int = 100,
    num_shots: int = 0,
    batch_size: int = 16,
    seed: int = 123,
) -> dict[str, float]:
    """调 lm_eval.simple_evaluate + MLXLM, 返回 {task_name: acc}。

    model: 已 fuse 的独立模型路径 (或 HF repo)。**不接受 adapter**。
    tasks: 具体任务名列表 (用 resolve_tasks 展开)。cmmlu 走本地, 其他走 hub (本周 hub 不通)。
    """
    import mlx.core as mx
    import lm_eval
    from mlx_lm.evaluate import MLXLM

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _install_cmmlu_local_patch()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mx.random.seed(seed)

    print("\n" + "=" * 60)
    print(f"评估: {model}")
    print(f"tasks: {tasks} | limit={limit} | num_shots={num_shots} | batch={batch_size}")
    print("=" * 60)

    lm = MLXLM(model, batch_size=batch_size, use_chat_template=False)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_shots,
        limit=limit,
        apply_chat_template=False,                       # raw completion (匹配 CPT 纯文本形式)
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )

    res = results.get("results", {})
    scores = _scores_from_results(res)

    # 落盘 (方便复用, 不重跑)
    tag = Path(str(model)).name.replace("/", "_") or "model"
    (output_dir / f"scores_{tag}.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return scores


# ─────────────────────────────────────────────
# 解析结果 (容错: 兼容 {task: metrics} 与 {results: {...}} 两种; 兼容多种 acc key)
# ─────────────────────────────────────────────
def _scores_from_results(results: dict) -> dict[str, float]:
    """从 results dict 提取 {task: acc} (容错 metric key)。"""
    if isinstance(results, dict) and "results" in results and isinstance(results["results"], dict):
        results = results["results"]
    scores: dict[str, float] = {}
    for task, metrics in results.items():
        acc = _find_acc(metrics)
        if acc is not None:
            scores[task] = acc
    return scores


def _find_acc(metrics) -> float | None:
    """从单个任务的 metrics 里挑准确率 (优先 acc, 兜底含 'acc' 不含 'stderr')。"""
    if not isinstance(metrics, dict):
        return None
    for k in ("acc", "acc,none", "acc_norm", "acc_norm,none"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    for k, v in metrics.items():
        if "acc" in k and "stderr" not in k and isinstance(v, (int, float)):
            return float(v)
    return None


def load_scores(output_dir: Path) -> dict[str, float]:
    """从 output_dir 读最新的 scores_*.json (run_mlx_evaluate 落盘的)。"""
    output_dir = Path(output_dir)
    jsons = sorted(output_dir.glob("scores_*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise FileNotFoundError(f"{output_dir} 下没找到 scores_*.json")
    return json.loads(jsons[-1].read_text(encoding="utf-8"))


def resolve_scores(arg, tasks, out_tag, limit=100, num_shots=0, batch_size=16, seed=123):
    """arg 是 .json (已有 {task: acc}) → 读; 否则当模型路径跑评估。

    out_tag: 输出子目录名 (base/cpt 用不同 tag 避免互相覆盖)。
    """
    p = Path(arg)
    if p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        return data
    out_dir = ROOT / "phase1" / "results" / "week12_eval" / out_tag
    return run_mlx_evaluate(arg, tasks, out_dir, limit, num_shots, batch_size, seed)


# ─────────────────────────────────────────────
# fuse: base + adapter → 独立模型 (evaluate 必须吃独立模型)
# ─────────────────────────────────────────────
def fuse_model(base_model: str, adapter_path: Path, save_path: Path) -> Path:
    """subprocess 跑 `python -m mlx_lm fuse ...`, 把全量微调产物合并成独立模型。

    全量微调 (fine-tune-type full) 的 fuse 是无损的 (依据 mlx-lm LORA.md)。
    """
    import subprocess
    import sys

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if (save_path / "config.json").exists():
        print(f"[fuse] 已存在, 跳过: {save_path}")
        return save_path

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", str(base_model),
        "--adapter-path", str(adapter_path),
        "--save-path", str(save_path),
    ]
    print("\n" + "=" * 60)
    print("Fuse 命令:")
    print(" ".join(cmd))
    print("=" * 60)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # mlx_lm fuse 末尾跑 create_model_card → ModelCard.load(), 离线时拉 README.md 会
        # 抛 OfflineModeIsEnabled/LocalEntryNotFoundError。但权重合并在那之前已完成 —— 只要
        # model.safetensors 落盘了, 这就是 cosmetic 失败, 当成功 (别让 README 卡住整个 eval)。
        # 真失败 (没落盘) 才 re-raise。
        if (save_path / "model.safetensors").exists():
            print(f"[fuse] ⚠ 子进程非零退出但 model.safetensors 已落盘 (card 步骤 cosmetic 失败), 当成功: {save_path}")
        else:
            raise
    print(f"[fuse] ✓ {save_path}")
    return save_path
