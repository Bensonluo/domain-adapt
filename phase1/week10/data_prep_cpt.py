"""
Phase 1 Week 10: CPT 数据准备脚本
================================

Tokenization + 混合比例配置 + 训练格式切分 + Token 统计报告。

设计原则 (用户偏好: 可切换 / 可扩展):
  - 数据源用 --source 开关切换，默认 synthetic (零下载，立即可跑)
  - tokenizer 用 --tokenizer 开关切换，无 transformers 时自动回退 FakeTokenizer

三种数据源:
  --source synthetic  : 合成医疗 / 通用文本样例 (默认, 无网络, 流程验证)
  --source local      : 读 phase1/data/raw/{domain,general}/ 本地语料 (训练前填充)
  --source hf         : HuggingFace datasets (留接口, 训练前接入)

tokenizer:
  --tokenizer Qwen/Qwen2.5-3B-Instruct  : 真实 Qwen tokenizer (离线读 HF 缓存, 需 transformers)
  --tokenizer fake                       : UTF-8 字节级 tokenizer (零依赖, 仅流程验证)

用法:
    # 默认: 合成数据 + 自动 tokenizer (有 transformers 用 Qwen, 否则 FakeTokenizer)
    python phase1/week10/data_prep_cpt.py

    # 强制零依赖跑通 (当前无 venv 时):
    python phase1/week10/data_prep_cpt.py --tokenizer fake

    # 真实数据 (训练前):
    python phase1/week10/data_prep_cpt.py --source local --tokenizer Qwen/Qwen2.5-3B-Instruct

思考锚点:
  - 为什么 CPT 必须用「目标模型」的 tokenizer?
    不同 tokenizer 的词表/切分粒度不同, 同一段文本 token 数可差 1.5-2x。
    用错 tokenizer 统计的 token 数 → 训练步数 / 显存预算全错。
  - overlap 有什么用? 设多少合适?
    避免 chunk 边界切断上下文, 但会重复学习边界 token。通常 0 (无重叠) 或小比例。
"""

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "phase1" / "data" / "processed" / "cpt_ready"

# ─────────────────────────────────────────────
# Tokenizer: 真实 Qwen (离线) + FakeTokenizer 回退
# ─────────────────────────────────────────────


class FakeTokenizer:
    """零依赖 UTF-8 字节级 tokenizer, 仅用于无 transformers 环境的流程验证。

    encode 按字节切分 → token id ∈ [0, 255], token 数 = UTF-8 字节数。
    真实训练必须换 Qwen tokenizer: 词表 ~150k, BPE 切分, token 数 ≠ 字节数。
    本类只保证 pipeline 逻辑 (混合 / 切块 / 统计) 可验证, 不代表真实 token 数。
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return 256

    def __repr__(self) -> str:
        return "FakeTokenizer(byte-level, vocab=256, DEMO ONLY)"


def load_tokenizer(name: str):
    """加载 tokenizer: fake → FakeTokenizer; 否则尝试 transformers 离线加载。

    离线策略: TRANSFORMERS_OFFLINE=1 → 只用本地 HF 缓存, 不联网。
    缓存里已有 Qwen/Qwen2.5-3B-Instruct (base 与 instruct 共享 tokenizer, CPT 无差异)。
    """
    if name == "fake":
        return FakeTokenizer()

    try:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        print(f"✓ 真实 tokenizer: {name} (vocab={len(tok)})")
        return tok
    except ImportError:
        print(
            "⚠️  transformers 未安装 → 回退 FakeTokenizer (仅流程验证, token 数不代表真实)。\n"
            "    要用真实 Qwen tokenizer: 建 phase1 venv → pip install transformers → 重跑。\n"
            "    模型已在 HF 缓存 (Qwen/Qwen2.5-3B-Instruct), 无需下载。"
        )
        return FakeTokenizer()
    except Exception as e:
        print(f"⚠️  加载 {name} 失败 ({type(e).__name__}: {e}) → 回退 FakeTokenizer")
        return FakeTokenizer()


# ─────────────────────────────────────────────
# 语料加载: synthetic / local / hf
# ─────────────────────────────────────────────


def _generate_synthetic_texts(
    domain: str, n: int, seed: int = 42
) -> list[str]:
    """生成领域 / 通用文本样例。

    真实数据留到训练前下载 (用户要求: 现在不下载)。
    合成数据只为跑通 pipeline + 验证 4 种配比逻辑, 不代表真实分布。
    """
    rng = random.Random(seed)

    # 医疗文本模板 (体现领域特征: 症状 / 诊断 / 药物 / 指南术语)
    medical_templates = [
        "患者{age}岁, 因{symptom}入院。查体: {sign}。初步诊断考虑{dx}, "
        "建议完善{test}并给予{drug}治疗, 注意监测{monitor}。",
        "{guideline}指南推荐: 对于{population}患者, {drug}可作为一线方案, "
        "起始剂量{dose}, 常见不良反应包括{ae}, 用药期间需定期复查{test}。",
        "病例分析: 该患者表现为{symptom}伴{sign}, 鉴别诊断需排除{ddx}。"
        "结合{test}结果, 支持{dx}诊断, 治疗上以{drug}为主, 辅以支持治疗。",
    ]
    pools = {
        "age": ["54", "67", "42", "71", "35", "58", "49"],
        "symptom": ["胸痛", "持续高热", "腹痛", "头晕伴呕吐", "呼吸困难", "关节肿痛"],
        "sign": ["血压 160/95", "心率 110 次/分", "腹部压痛", "病理征阳性", "双肺湿啰音"],
        "dx": ["急性心肌梗死", "社区获得性肺炎", "急性阑尾炎", "脑梗死", "心力衰竭"],
        "test": ["心电图", "血常规", "CT 检查", "心肌酶谱", "腹部超声"],
        "drug": ["阿司匹林", "头孢曲松", "硝酸甘油", "低分子肝素", "利尿剂"],
        "monitor": ["出凝血时间", "肝肾功能", "电解质", "心电图变化", "尿量"],
        "guideline": ["NCCN", "CSCO", "ESC", "中国高血压"],
        "population": ["老年高血压", "糖尿病合并冠心", "慢性肾病", "急性缺血性卒中"],
        "dose": ["75mg qd", "1g qd", "0.4mg 舌下含服", "4000U 皮下注射 q12h"],
        "ae": ["出血风险", "胃肠道反应", "低血压", "肝功能异常", "皮疹"],
        "ddx": ["肺栓塞", "主动脉夹层", "消化道穿孔", "急性胰腺炎"],
    }

    # 通用文本模板 (百科 / 新闻风格, 与领域分布明显不同)
    general_templates = [
        "{city}是一座位于{region}的{attr}城市, 以{feature}闻名, "
        "每年吸引大量游客前来{activity}, 当地{food}也颇具特色。",
        "近日, {org}发布报告称, 今年{field}行业{trend}, "
        "专家认为这与{factor}密切相关, 预计未来仍将{forecast}。",
        "{concept}是{subject}中的一个重要概念, 指的是{definition}, "
        "它最早由{person}提出, 对后来的{impact}产生了深远影响。",
    ]
    g_pools = {
        "city": ["杭州", "成都", "西安", "苏州", "青岛", "厦门"],
        "region": ["华东", "西南", "西北", "江南", "胶东半岛"],
        "attr": ["历史", "宜居", "港口", "旅游", "文化"],
        "feature": ["西湖", "美食", "古城墙", "园林", "海岸线"],
        "activity": ["观光", "度假", "体验", "考察"],
        "food": ["小吃", "菜系", "特产", "海鲜"],
        "org": ["研究院", "行业协会", "咨询机构"],
        "field": ["新能源", "半导体", "人工智能", "消费电子"],
        "trend": ["保持增长", "出现分化", "加速整合"],
        "factor": ["政策支持", "技术突破", "需求回暖"],
        "forecast": ["稳步发展", "持续向好", "保持韧性"],
        "concept": ["相对论", "边际效用", "光合作用", "社会契约"],
        "subject": ["物理学", "经济学", "生物学", "政治学"],
        "definition": ["时空的几何关系", "递增规律", "能量转化", "权利让渡"],
        "person": ["爱因斯坦", "边际学派", "科学家", "启蒙思想家"],
        "impact": ["理论发展", "学科演进", "工业应用", "制度设计"],
    }

    templates = medical_templates if domain == "domain" else general_templates
    p = pools if domain == "domain" else g_pools

    texts = []
    for _ in range(n):
        tpl = rng.choice(templates)
        filled = tpl.format(**{k: rng.choice(v) for k, v in p.items()})
        # 拼接几条成一段, 让单条文本有一定长度
        texts.append(filled)
    return texts


def load_corpus(source: str, corpus_dir: str) -> tuple[list[str], list[str]]:
    """加载 (领域文本, 通用文本)。三路分发, 返回两个文本列表。"""
    if source == "synthetic":
        # 合成: 默认各 120 条, 总 token 足够切出数十个 chunk
        print("[source=synthetic] 生成医疗 / 通用样例 (零下载, 仅流程验证)")
        domain_texts = _generate_synthetic_texts("domain", n=120)
        general_texts = _generate_synthetic_texts("general", n=120)
        return domain_texts, general_texts

    if source == "local":
        # 本地: corpus_dir/{domain,general}/*.txt (训练前填充真实语料)
        base = Path(corpus_dir)
        domain_texts = _read_text_dir(base / "domain")
        general_texts = _read_text_dir(base / "general")
        if not domain_texts:
            raise FileNotFoundError(
                f"本地领域语料为空: {base/'domain'}。"
                f"训练前请把医疗语料放进去 (每文件一条或多条文本)。"
            )
        print(f"[source=local] 领域 {len(domain_texts)} 条 / 通用 {len(general_texts)} 条")
        return domain_texts, general_texts

    if source == "hf":
        # HF 接口: 训练前接入 (PubMed 摘要 / Wikipedia 中文, 经 hf-mirror)
        raise NotImplementedError(
            "HF 源留待训练前接入: datasets.load_dataset(...) 经 HF_ENDPOINT=https://hf-mirror.com。"
            " 用户已确认: 现在不下载, 训练前再接。"
        )

    raise ValueError(f"未知 source: {source} (可选 synthetic|local|hf)")


def _read_text_dir(dir_path: Path) -> list[str]:
    """读取目录下所有 .txt 文件, 每个非空行作为一条文本。"""
    if not dir_path.exists():
        return []
    texts = []
    for f in sorted(dir_path.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if len(line) > 50:  # 与 clean_pipeline 的长度阈值一致
                texts.append(line)
    return texts


# ─────────────────────────────────────────────
# 统计 + 混合 (脚手架保留 + 增强)
# ─────────────────────────────────────────────


def count_tokens(texts: list[str], tokenizer) -> int:
    """统计文本列表的总 token 数。"""
    total = 0
    for text in texts:
        total += len(tokenizer.encode(text))
    return total


def mix_data(domain_texts: list[str], general_texts: list[str], ratio: str) -> list[str]:
    """按「文档数」比例混合 (脚手架原始实现, 快速近似)。

    注意: 文档数比例 ≠ token 比例。若领域文档普遍更长, 70-30 (文档) 的实际 token
    比例可能变成 85-15。主流程用 mix_token_streams 按 token 精确混合, 本函数保留供对比。
    """
    domain_pct, general_pct = map(int, ratio.split("-"))
    total = domain_pct + general_pct
    domain_n = int(len(domain_texts) * domain_pct / total)
    general_n = int(len(general_texts) * general_pct / total)
    return domain_texts[:domain_n] + general_texts[:general_n]


def mix_token_streams(
    domain_ids: list[int], general_ids: list[int], ratio: str
) -> tuple[list[int], dict]:
    """按「token 数」精确混合两个 token 流, 返回 (混合 token 流, 统计)。

    策略: 以目标比例 dp:gp 取, 受限于较小的语料 (按比例取到某个耗尽为止)。
    这样混合后的实际 token 比例尽量贴近目标, 优于按文档数近似。
    """
    dp, gp = map(int, ratio.split("-"))

    # 纯领域: 全用 domain, 不掺 general
    if gp == 0:
        mixed = list(domain_ids)
        stats = {
            "ratio_target": ratio,
            "domain_tokens": len(domain_ids),
            "general_tokens": 0,
            "actual_domain_ratio": 1.0,
        }
        return mixed, stats

    # 全用 domain 时需要的 general 量; 若够 → domain 是基准
    need_general_if_full_domain = len(domain_ids) * gp / dp
    if need_general_if_full_domain <= len(general_ids):
        d_take = len(domain_ids)
        g_take = int(need_general_if_full_domain)
    else:
        # general 是基准, 全用 general, domain 按比例取
        g_take = len(general_ids)
        d_take = int(len(general_ids) * dp / gp)

    mixed = domain_ids[:d_take] + general_ids[:g_take]
    total = len(mixed)
    stats = {
        "ratio_target": ratio,
        "domain_tokens": d_take,
        "general_tokens": g_take,
        "actual_domain_ratio": round(d_take / max(total, 1), 3),
    }
    return mixed, stats


def pack_into_chunks(
    token_stream: list[int], seq_length: int, overlap: int = 0
) -> list[list[int]]:
    """把扁平 token 流切成固定长度 chunk (CPT 训练格式)。

    overlap: 相邻 chunk 重叠的 token 数。0 = 无重叠 (最常用)。
      overlap > 0 可避免文档边界处的上下文丢失, 但边界 token 会被重复学习。
      经验: overlap 一般 ≤ seq_length * 0.1。
    末尾不足 seq_length 的尾部丢弃 (CPT 不做 padding, 避免 pad token 污染)。
    """
    if seq_length <= 0:
        raise ValueError(f"seq_length 必须 > 0, 得到 {seq_length}")
    if overlap < 0 or overlap >= seq_length:
        raise ValueError(f"overlap 须 ∈ [0, seq_length), 得到 {overlap}")

    step = seq_length - overlap
    chunks = []
    for start in range(0, len(token_stream) - seq_length + 1, step):
        chunks.append(token_stream[start : start + seq_length])
    return chunks


# ─────────────────────────────────────────────
# 保存 + 报告
# ─────────────────────────────────────────────


@dataclass
class RatioReport:
    """单个配比的统计结果 (可序列化为 json)。"""

    ratio: str
    tokenizer: str
    seq_length: int
    overlap: int
    n_chunks: int
    total_tokens: int
    domain_tokens: int
    general_tokens: int
    actual_domain_ratio: float
    target_domain_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def save_chunks_jsonl(chunks: list[list[int]], tokenizer, path: Path) -> None:
    """保存 chunk 列表为 jsonl。

    每行: {"text": 解码文本, "ids": token id 列表, "n_tokens": chunk 长度}
    兼容 week3/week11 消费: 用 "text" 字段 (Dataset.from_json + 在线 tokenize)。
    真实 token 在 "ids" 字段 (避免 detokenize→retokenize 的边界偏差)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ids in chunks:
            f.write(
                json.dumps(
                    {
                        "text": tokenizer.decode(ids),
                        "ids": ids,
                        "n_tokens": len(ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CPT 数据准备: tokenize + 混合 + 切块 + 报告")
    parser.add_argument("--corpus", default="phase1/data/processed/cpt/", help="本地语料目录 (source=local 时用)")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "local", "hf"],
                        help="数据源 (默认 synthetic 零下载)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct",
                        help="tokenizer (fake = 零依赖验证; 真实名 = 离线读缓存)")
    parser.add_argument("--ratios", default="100-0,70-30,50-50,30-70",
                        help="混合比例 domain-general, 逗号分隔")
    parser.add_argument("--seq-length", type=int, default=2048, help="每个 chunk 的 token 长度")
    parser.add_argument("--overlap", type=int, default=0, help="chunk 间重叠 token 数 (0=无重叠)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 tokenizer (真实 Qwen 离线 / FakeTokenizer 回退)
    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer_name = args.tokenizer

    # 2. 加载领域 + 通用语料
    domain_texts, general_texts = load_corpus(args.source, args.corpus)
    print(f"领域语料: {len(domain_texts)} 条 ({count_tokens(domain_texts, tokenizer):,} tokens)")
    print(f"通用语料: {len(general_texts)} 条 ({count_tokens(general_texts, tokenizer):,} tokens)")

    # 文本 → token 流 (一次性 tokenize, 后续按 token 精确混合)
    domain_ids = [t for text in domain_texts for t in tokenizer.encode(text)]
    general_ids = [t for text in general_texts for t in tokenizer.encode(text)]
    print(f"领域 token 流: {len(domain_ids):,} | 通用 token 流: {len(general_ids):,}")

    # 3. 对每种比例: 按比例混合 → 切固定长度 chunk → 保存 jsonl
    ratios = args.ratios.split(",")
    reports = []
    for ratio in ratios:
        ratio = ratio.strip()
        mixed_ids, mix_stats = mix_token_streams(domain_ids, general_ids, ratio)
        chunks = pack_into_chunks(mixed_ids, args.seq_length, args.overlap)

        out_file = output_dir / f"cpt_{ratio}.jsonl"
        save_chunks_jsonl(chunks, tokenizer, out_file)

        dp = int(ratio.split("-")[0])
        report = RatioReport(
            ratio=ratio,
            tokenizer=tokenizer_name,
            seq_length=args.seq_length,
            overlap=args.overlap,
            n_chunks=len(chunks),
            total_tokens=len(chunks) * args.seq_length,
            domain_tokens=mix_stats["domain_tokens"],
            general_tokens=mix_stats["general_tokens"],
            actual_domain_ratio=mix_stats["actual_domain_ratio"],
            target_domain_ratio=round(dp / sum(map(int, ratio.split("-"))), 3),
        )
        reports.append(report.to_dict())
        print(
            f"  ratio {ratio:>6}: {len(chunks):>4} chunks | "
            f"token 比 实际 {mix_stats['actual_domain_ratio']:.2f} / 目标 {report.target_domain_ratio:.2f} | "
            f"→ {out_file.name}"
        )

    # 4. 输出统计报告 (与 week11 train_cpt.py 的 max_steps 对照)
    report_path = output_dir / "token_report.json"
    summary = {
        "tokenizer": tokenizer_name,
        "source": args.source,
        "seq_length": args.seq_length,
        "overlap": args.overlap,
        "domain_tokens_available": len(domain_ids),
        "general_tokens_available": len(general_ids),
        "configs": reports,
        "note": (
            "FakeTokenizer 下 token 数 = UTF-8 字节数, 仅供流程验证。"
            "换 Qwen tokenizer 后重跑得到真实 token 数。"
            if tokenizer_name == "fake" or isinstance(tokenizer, FakeTokenizer)
            else "真实 Qwen tokenizer 统计。"
        ),
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Token 报告: {report_path}")
    print(f"✓ {len(reports)} 种配比已就绪: {output_dir}/cpt_*.jsonl")
    print("\n下一步 (week11): 用某个 cpt_{ratio}.jsonl 作为 --data 跑 train_cpt.py")


if __name__ == "__main__":
    main()
