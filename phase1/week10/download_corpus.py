"""
Phase 1 Week 10: 真实 CPT 语料下载 (魔搭 ModelScope)
=====================================================

把 week10 的 --source local 接口接到真实中文语料:
  - 领域 (医疗): zjydiary/Medical (shibing624/medical 魔搭镜像) 的
                pretrain/medical_book_zh.json — 中文医学百科纯文本 (CPT 正确形态)
  - 通用:        AI-ModelScope/wikipedia-cn-20230720-filtered 的
                wikipedia-cn-20230720-filtered.jsonl — 中文维基纯文本

落地为 phase1/data/raw/{domain,general}/*.txt, 供 data_prep_cpt.py --source local 消费。

为什么用 snapshot_download 而不是 MsDataset.load:
  - zjydiary/Medical 的 default subset 是 QA (instruction/input/output, SFT 数据),
    CPT 要的纯文本在 pretrain/ 子目录的原始 .json, 不在 default subset 暴露
  - wikipedia 源没有加载脚本, MsDataset.load 会 RecursionError
  - 两源都是标准 jsonl, 下原始文件本地解析最可控 (见 [[modelscope-qwen-download]])

用法:
    # 先验证领域源 (40MB, 快)
    python phase1/week10/download_corpus.py --sources domain --max-domain 10000

    # 再下通用源 (520MB, 慢些)
    python phase1/week10/download_corpus.py --sources general --max-general 10000

    # 两源都下 (默认)
    python phase1/week10/download_corpus.py

    # 下完跑数据准备:
    python phase1/week10/data_prep_cpt.py --source local --corpus phase1/data/raw \
        --tokenizer models/Qwen3.5-0.8B-Base-ms
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "phase1" / "data" / "raw"

# ─────────────────────────────────────────────
# 源配置 (加新源只加一行, 可切换/可扩展)
# ─────────────────────────────────────────────
SOURCES = {
    "domain": {
        "repo": "zjydiary/Medical",
        # CPT 纯文本在 pretrain/ 下; medical_book_zh 最纯 (40MB), 不够再补 train_encyclopedia (591MB)
        "files": ["pretrain/medical_book_zh.json"],
        "out_subdir": "domain",
        "out_name": "medical.txt",
    },
    "general": {
        "repo": "AI-ModelScope/wikipedia-cn-20230720-filtered",
        "files": ["wikipedia-cn-20230720-filtered.jsonl"],
        "out_subdir": "general",
        "out_name": "wiki.txt",
    },
}


# ─────────────────────────────────────────────
# 文本提取 (容错: 不依赖精确字段名)
# ─────────────────────────────────────────────
TEXT_KEYS = ("text", "completion", "content", "sentence", "body", "raw")


def extract_text(obj) -> str:
    """从一条 json 对象里提取纯文本。容错多种 schema:
    - 字符串直接返回
    - dict: 优先 text/completion/content 等字段; 维基 {title,text} 拼接; 兜底最长 string
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in TEXT_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # 维基常见 {title, text}: 拼标题 + 正文
        parts = [obj.get(k) for k in ("title", "text") if isinstance(obj.get(k), str) and obj.get(k).strip()]
        if parts:
            return "\n".join(parts)
        # 兜底: 取最长的 string 值
        strs = [v for v in obj.values() if isinstance(v, str)]
        if strs:
            return max(strs, key=len)
    return ""


def iter_json_records(path: Path):
    """迭代 jsonl 文件的每条记录。若整文件是单个 json array 则按数组迭代。"""
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        first_stripped = first.strip()
        # 单个 json array 的情形: 第一行以 '[' 开头
        if first_stripped == "[" or first_stripped.startswith("[{"):
            data = json.loads(first + f.read())
            for rec in data:
                yield rec
            return
        # 否则按 jsonl: 第一行也是一条
        if first_stripped:
            try:
                yield json.loads(first_stripped)
            except json.JSONDecodeError:
                pass
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def collect_texts(path: Path, max_n: int, min_len: int = 50) -> list[str]:
    """从 jsonl 文件收集 ≤ max_n 条纯文本 (长度 > min_len, 去空)。"""
    texts: list[str] = []
    for rec in iter_json_records(path):
        if len(texts) >= max_n:
            break
        t = extract_text(rec).strip()
        if len(t) > min_len:
            texts.append(t)
    return texts


def write_texts(texts: list[str], out_file: Path) -> None:
    """每条文本写一行 (内部换行/多空格压缩为单空格, 与 _read_text_dir 的「行=一条」对齐)。"""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(" ".join(t.split()) + "\n")


# ─────────────────────────────────────────────
# 下载 (snapshot_download 下原始 LFS 文件)
# ─────────────────────────────────────────────
def download_source(src_key: str, max_n: int, corpus_dir: Path, cache_dir: Path) -> Path:
    """下某个源的 jsonl 文件, 解析, 落地到 corpus_dir/{out_subdir}/{out_name}。返回输出路径。"""
    from modelscope import snapshot_download

    cfg = SOURCES[src_key]
    out_file = corpus_dir / cfg["out_subdir"] / cfg["out_name"]

    # 幂等: 已有且行数达标则跳过下载
    if out_file.exists() and sum(1 for _ in out_file.open(encoding="utf-8")) >= max_n:
        print(f"[{src_key}] 已存在且 ≥ {max_n} 条, 跳过: {out_file}")
        return out_file

    print(f"[{src_key}] 下载 {cfg['repo']} / {cfg['files']} → 缓存 {cache_dir}")
    local_path = snapshot_download(
        cfg["repo"],
        repo_type="dataset",
        allow_patterns=cfg["files"],
        cache_dir=str(cache_dir),
    )
    base = Path(local_path)

    texts: list[str] = []
    for rel in cfg["files"]:
        f = base / rel
        if not f.exists():
            print(f"  ⚠️ 未找到 {f}, 跳过")
            continue
        need = max_n - len(texts)
        if need <= 0:
            break
        got = collect_texts(f, need)
        print(f"  {rel}: 取 {len(got)} 条 (累计 {len(texts) + len(got)})")
        # 打印首条样本确认字段提取对
        if not texts and got:
            preview = " ".join(got[0].split())[:120]
            print(f"  样本[0]: {preview}...")
        texts.extend(got)

    if not texts:
        raise RuntimeError(f"[{src_key}] 未取到任何文本, 检查源/字段")

    write_texts(texts, out_file)
    print(f"[{src_key}] ✓ 写入 {len(texts)} 条 → {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="下载真实中文 CPT 语料 (魔搭) → phase1/data/raw/")
    parser.add_argument("--sources", default="both", choices=["domain", "general", "both"])
    parser.add_argument("--max-domain", type=int, default=10000, help="领域 (医疗) 条数上限")
    parser.add_argument("--max-general", type=int, default=10000, help="通用 (维基) 条数上限")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="输出根目录")
    parser.add_argument("--cache", default=str(ROOT / "phase1" / "data" / ".ms_cache"),
                        help="魔搭下载缓存 (复用, 避免 re-download)")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    if args.sources in ("domain", "both"):
        todo.append(("domain", args.max_domain))
    if args.sources in ("general", "both"):
        todo.append(("general", args.max_general))

    for key, mx in todo:
        download_source(key, mx, corpus_dir, cache_dir)

    print("\n下一步: 用真实 tokenizer 重跑数据准备")
    print("  python phase1/week10/data_prep_cpt.py --source local "
          f"--corpus {args.corpus} --tokenizer models/Qwen3.5-0.8B-Base-ms")


if __name__ == "__main__":
    main()
