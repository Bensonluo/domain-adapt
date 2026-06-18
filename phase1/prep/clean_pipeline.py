"""
Phase 1 Prep: 数据清洗 Pipeline
================================

领域无关的清洗管线, 换领域只改配置 (DOMAIN_CONFIGS)。

清洗步骤:
  1. _remove_boilerplate  去 HTML 标签 / 模板 / 多余空行
  2. _length_filter        过滤过短 / 过长
  3. _dedup_exact          精确去重
  4. _dedup_minhash        MinHash 近似去重 (需 datasketch, 缺失则跳过)

依赖:
  - datasketch 可选 (MinHash 去重)。未装时自动跳过该步, 其余清洗照常。

用法:
    # 零依赖演示 (无 venv 时验证清洗逻辑):
    python phase1/prep/clean_pipeline.py --config medical --source synthetic

    # 真实数据 (训练前):
    python phase1/prep/clean_pipeline.py --config medical --input phase1/data/raw/medical/
"""

import json
import random
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "phase1" / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "phase1" / "data" / "processed"

# ─────────────────────────────────────────────
# 领域配置: 换领域只改这里 (可插拔)
# ─────────────────────────────────────────────

DOMAIN_CONFIGS = {
    "medical": {
        "min_length": 100,
        "max_length": 50000,
        "minhash_threshold": 0.8,
    },
    "finance": {
        "min_length": 120,
        "max_length": 50000,
        "minhash_threshold": 0.85,
    },
    "general": {
        "min_length": 80,
        "max_length": 50000,
        "minhash_threshold": 0.8,
    },
}


class DataCleaningPipeline:
    """通用数据清洗管线 (领域无关, 配置驱动)。"""

    def __init__(self, config: dict):
        self.config = config
        self.stats = {"input": 0, "output": 0, "dropped": {}}

    def run(self, texts: list[str]) -> list[str]:
        self.stats["input"] = len(texts)

        texts = self._remove_boilerplate(texts)
        texts = self._length_filter(texts)
        texts = self._dedup_exact(texts)
        texts = self._dedup_minhash(texts)
        # TODO(进阶): 实现 _ppl_filter — 用基座模型困惑度过滤
        # TODO(进阶): 实现 _domain_relevance_filter — 关键词 + embedding

        self.stats["output"] = len(texts)
        return texts

    def _remove_boilerplate(self, texts: list[str]) -> list[str]:
        """去 HTML 标签、模板文字、重复空行。"""
        cleaned = []
        for t in texts:
            t = re.sub(r"<[^>]+>", "", t)          # HTML 标签
            t = re.sub(r"\n{3,}", "\n\n", t)        # 多余空行
            t = t.strip()
            if len(t) > 50:                         # 顺手挡掉残留短碎片
                cleaned.append(t)
        self.stats["dropped"]["boilerplate"] = len(texts) - len(cleaned)
        return cleaned

    def _length_filter(self, texts: list[str]) -> list[str]:
        """过滤过短 / 过长文本。"""
        min_len = self.config.get("min_length", 100)
        max_len = self.config.get("max_length", 50000)
        filtered = [t for t in texts if min_len <= len(t) <= max_len]
        self.stats["dropped"]["length"] = len(texts) - len(filtered)
        return filtered

    def _dedup_exact(self, texts: list[str]) -> list[str]:
        """精确去重 (完全相同的文本)。"""
        seen = set()
        result = []
        for t in texts:
            key = t.strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(t)
        self.stats["dropped"]["exact_dedup"] = len(texts) - len(result)
        return result

    def _dedup_minhash(
        self, texts: list[str], threshold: float | None = None, num_perm: int = 128
    ) -> list[str]:
        """MinHash 近似去重 (需 datasketch; 未安装则跳过)。"""
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            print("⚠️  datasketch 未安装 → 跳过 MinHash 近似去重 (pip install datasketch 启用)")
            self.stats["dropped"]["minhash_dedup"] = 0
            self.stats["dropped"]["minhash_skipped"] = True
            return texts

        if threshold is None:
            threshold = self.config.get("minhash_threshold", 0.8)
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        result = []
        for i, t in enumerate(texts):
            mh = MinHash(num_perm=num_perm)
            for word in t.split():
                mh.update(word.encode("utf-8"))
            if not lsh.query(mh):           # 没有近似重复 → 保留
                lsh.insert(str(i), mh)
                result.append(t)
        self.stats["dropped"]["minhash_dedup"] = len(texts) - len(result)
        return result

    def get_stats(self) -> dict:
        total_dropped = self.stats["input"] - self.stats["output"]
        return {
            "input": self.stats["input"],
            "output": self.stats["output"],
            "drop_rate": round(total_dropped / max(self.stats["input"], 1), 3),
            "breakdown": self.stats["dropped"],
        }


# ─────────────────────────────────────────────
# 数据源: synthetic (带脏数据演示) / 本地 raw
# ─────────────────────────────────────────────


def generate_synthetic_raw(domain: str, n: int = 200, seed: int = 42) -> list[str]:
    """生成带「故意脏数据」的原始语料, 用于演示/验证清洗逻辑。

    混入: HTML 标签 / 过短碎片 / 完全重复 / 近似重复 / 正常文本。
    真实数据留到训练前下载 (用户要求: 现在不下载)。
    """
    rng = random.Random(seed)
    clean_pool = [
        "患者主诉胸闷气促三天, 加重伴大汗一小时入院。既往高血压病史十年, "
        "规律服用降压药物。查体血压偏高, 心率增快, 听诊双肺底可闻及湿性啰音, "
        "心电图提示 ST 段抬高, 心肌酶谱明显升高, 考虑急性心肌梗死, 立即启动急诊介入流程。",
        "社区获得性肺炎的常见病原体包括肺炎链球菌、流感嗜血杆菌等, 经验性抗感染治疗 "
        "可选用 beta 内酰胺类联合大环内酯类。治疗 48 至 72 小时后需评估临床反应, "
        "若体温下降、呼吸频率改善、氧合稳定提示治疗有效, 可考虑序贯口服治疗。",
        "2 型糖尿病的综合管理包括血糖控制、血压管理、血脂调节及抗血小板治疗。"
        "糖化血红蛋白目标值一般控制在百分之七以下, 但需根据患者年龄、合并症个体化调整, "
        "老年或合并多种疾病的患者可适当放宽目标, 避免低血糖事件发生。",
    ]

    texts = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.5:
            # 正常医疗文本 (有时拼两条增加长度)
            base = rng.choice(clean_pool)
            if rng.random() < 0.3:
                base += rng.choice(clean_pool)
            texts.append(base)
        elif kind < 0.65:
            # 带 HTML 标签 (测 _remove_boilerplate)
            tpl = rng.choice(clean_pool)
            texts.append(f"<p>{tpl}</p>")
        elif kind < 0.75:
            # 过短碎片 (测 _length_filter)
            texts.append(rng.choice(["短", "test 123", "嗯", "短文本"]))
        elif kind < 0.85:
            # 完全重复 (测 _dedup_exact)
            texts.append(clean_pool[0])
        else:
            # 近似重复: 在原文基础上小改 (测 _dedup_minhash)
            base = clean_pool[0].replace("急性", "亚急性").replace("立刻", "马上")
            texts.append(base)
    return texts


def read_raw_dir(input_dir: str) -> list[str]:
    """读取目录下所有 .txt 文件, 每个非空行作为一条原始文本。"""
    base = Path(input_dir)
    if not base.exists():
        return []
    texts = []
    for f in sorted(base.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def save_cleaned_jsonl(texts: list[str], path: Path) -> None:
    """保存清洗后文本为 jsonl, 每行 {"text": ...}。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="数据清洗管线 (领域可插拔)")
    parser.add_argument("--config", default="medical", help="领域配置名 (medical/finance/general)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="原始语料目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出目录")
    parser.add_argument("--source", default=None, choices=[None, "synthetic"],
                        help="无真实数据时用 synthetic 演示清洗逻辑")
    parser.add_argument("--n-synthetic", type=int, default=200, help="synthetic 模式生成条数")
    args = parser.parse_args()

    # 1. 加载领域配置 (换领域只改 DOMAIN_CONFIGS)
    config = DOMAIN_CONFIGS.get(args.config, DOMAIN_CONFIGS["medical"])
    print(f"[config={args.config}] {config}")

    # 2. 读取原始数据
    if args.source == "synthetic":
        print(f"[source=synthetic] 生成 {args.n_synthetic} 条带脏数据的演示语料 (零下载)")
        texts = generate_synthetic_raw(args.config, args.n_synthetic)
    else:
        texts = read_raw_dir(args.input)
        if not texts:
            print(
                f"原始语料为空: {args.input}\n"
                f"  → 放 .txt 文件进去 (每行一条), 或用 --source synthetic 演示清洗逻辑。"
            )
            return
        print(f"[source=local] 读取 {len(texts)} 条原始文本")

    # 3. 运行清洗管线
    pipeline = DataCleaningPipeline(config)
    cleaned = pipeline.run(texts)

    # 4. 保存清洗后数据
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_file = out_dir / f"{args.config}_cleaned.jsonl"
    save_cleaned_jsonl(cleaned, cleaned_file)

    # 5. 输出清洗统计报告
    stats = pipeline.get_stats()
    report_file = out_dir / f"{args.config}_cleaning_report.json"
    report_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印摘要
    print(f"\n输入: {stats['input']} → 输出: {stats['output']} (drop_rate={stats['drop_rate']})")
    print("各步丢弃:")
    for step, n in stats["breakdown"].items():
        print(f"  {step:>16}: {n}")
    print(f"\n✓ 清洗后语料: {cleaned_file}")
    print(f"✓ 清洗报告: {report_file}")


if __name__ == "__main__":
    main()
