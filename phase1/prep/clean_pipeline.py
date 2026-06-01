"""
Phase 1 Prep: 数据清洗 Pipeline

领域无关的清洗管线，换领域只改配置。

Usage:
    python phase1/prep/clean_pipeline.py --config medical
"""

import re
import json
import argparse
from pathlib import Path
from datasketch import MinHash, MinHashLSH


class DataCleaningPipeline:
    """通用数据清洗管线"""

    def __init__(self, config: dict):
        self.config = config
        self.stats = {"input": 0, "output": 0, "dropped": {}}

    def run(self, texts: list[str]) -> list[str]:
        self.stats["input"] = len(texts)

        texts = self._remove_boilerplate(texts)
        texts = self._length_filter(texts)
        texts = self._dedup_exact(texts)
        texts = self._dedup_minhash(texts)
        # TODO: 实现 _ppl_filter — 用基座模型困惑度过滤
        # TODO: 实现 _domain_relevance_filter — 关键词 + embedding

        self.stats["output"] = len(texts)
        return texts

    def _remove_boilerplate(self, texts: list[str]) -> list[str]:
        """去 HTML 标签、模板文字、页眉页脚、重复空行"""
        cleaned = []
        for t in texts:
            t = re.sub(r"<[^>]+>", "", t)
            t = re.sub(r"\n{3,}", "\n\n", t)
            t = t.strip()
            if len(t) > 50:
                cleaned.append(t)
        self.stats["dropped"]["boilerplate"] = len(texts) - len(cleaned)
        return cleaned

    def _length_filter(self, texts: list[str]) -> list[str]:
        """过滤过短/过长文本"""
        min_len = self.config.get("min_length", 100)
        max_len = self.config.get("max_length", 50000)
        filtered = [t for t in texts if min_len <= len(t) <= max_len]
        self.stats["dropped"]["length"] = len(texts) - len(filtered)
        return filtered

    def _dedup_exact(self, texts: list[str]) -> list[str]:
        """精确去重"""
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
        self, texts: list[str], threshold: float = 0.8, num_perm: int = 128
    ) -> list[str]:
        """MinHash 近似去重"""
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        result = []
        for i, t in enumerate(texts):
            mh = MinHash(num_perm=num_perm)
            for word in t.split():
                mh.update(word.encode("utf-8"))
            if not lsh.query(mh):
                lsh.insert(str(i), mh)
                result.append(t)
        self.stats["dropped"]["minhash_dedup"] = len(texts) - len(result)
        return result

    def get_stats(self) -> dict:
        total_dropped = self.stats["input"] - self.stats["output"]
        return {
            "input": self.stats["input"],
            "output": self.stats["output"],
            "drop_rate": total_dropped / max(self.stats["input"], 1),
            "breakdown": self.stats["dropped"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medical", help="Domain config name")
    parser.add_argument("--input", default="phase1/data/raw/", help="Input directory")
    parser.add_argument("--output", default="phase1/data/processed/", help="Output directory")
    args = parser.parse_args()

    # TODO: 加载领域配置
    config = {"min_length": 100, "max_length": 50000}

    # TODO: 读取原始数据
    # TODO: 运行清洗管线
    # TODO: 保存清洗后数据
    # TODO: 输出清洗统计报告

    print("TODO: Implement main pipeline")


if __name__ == "__main__":
    main()
