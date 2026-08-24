"""Resumable Evol-Instruct pipeline for accepted Week 21 medical MCQs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import (
    MLXGenerator,
    append_jsonl,
    max_similarity,
    ngrams,
    normalize_mcq,
    parse_json_collection,
    read_jsonl,
    stable_id,
    write_jsonl,
)

EVOL_PROMPTS = {
    "constraints": "加入一个临床约束（年龄、合并症、禁忌证或检验条件），使作答必须同时满足多个条件。",
    "deepening": "提高推理深度，要求先识别关键证据，再进行鉴别诊断或机制判断。",
    "concretizing": "把抽象问题改成具体病例，提供必要且不泄露答案的症状、体征或检查结果。",
    "reasoning": "改写为需要两步以上医学推理的问题，但仍保持唯一最佳答案。",
}
SYSTEM = (
    "你是中国医学考试命题专家。只输出合法 JSON，不输出 Markdown 或思考过程。"
    "演化后必须保持医学正确、单项选择、唯一最佳答案。"
)


def output_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix == ".jsonl":
        return output, output.with_name("evol_instruct_raw.jsonl"), output.with_name("evol_instruct_stats.json")
    return output / "evol_instruct.jsonl", output / "evol_instruct_raw.jsonl", output / "evol_instruct_stats.json"


def build_prompt(source: dict[str, Any], strategies: list[str]) -> str:
    source_json = json.dumps({key: source[key] for key in ("question", "options", "answer", "explanation")}, ensure_ascii=False)
    instructions = "\n".join(f"第 {index + 1} 步（{name}）：{EVOL_PROMPTS[name]}" for index, name in enumerate(strategies))
    return f"""原题：{source_json}

依次完成 {len(strategies)} 步演化：
{instructions}

返回 JSON 对象，恰含 stages 数组。stages 中每一步是完整题目并含 question/options/answer/explanation 四个字段。
不得只在题干末尾添加无关文字；每步必须实质提高复杂度；答案必须与新题及解析一致。
为避免截断，每一步必须遵守：question 不超过 150 个汉字；每个 option value 不超过 30 个汉字；
explanation 不超过 100 个汉字，只解释关键证据和正确项。输出紧凑 JSON，不要缩进。"""


def mock_stages(source: dict[str, Any], strategies: list[str]) -> list[dict[str, Any]]:
    current = {key: source[key] for key in ("question", "options", "answer", "explanation")}
    stages = []
    additions = {
        "constraints": "若患者同时存在肾功能不全，",
        "deepening": "结合病理生理机制与鉴别诊断，",
        "concretizing": "一名门诊患者出现相关症状和典型检查结果，",
        "reasoning": "先判断关键证据，再选择处理方案：",
    }
    for name in strategies:
        current = {**current, "question": additions[name] + current["question"]}
        stages.append(current)
    return stages


def accepted_record(raw: dict[str, Any], source: dict[str, Any], threshold: float):
    parsed = raw.get("parsed")
    if not isinstance(parsed, list) or not parsed:
        parsed = parse_json_collection(str(raw.get("response", "")))
    target_depth = int(raw["target_depth"])
    if len(parsed) < target_depth:
        return None, "insufficient_stages"
    stages_out = []
    previous = source
    for depth_index, candidate in enumerate(parsed[:target_depth], 1):
        normalized, error = normalize_mcq(candidate)
        if error:
            return None, error
        similarity = max_similarity(normalized["question"], [ngrams(previous["question"])])
        if normalized["question"] == previous["question"] or similarity >= threshold:
            return None, "insufficient_evolution"
        stages_out.append({
            "depth": depth_index, "strategy": raw["strategies"][depth_index - 1], **normalized
        })
        previous = normalized
    final = stages_out[-1]
    return {
        "synthetic_id": f"evol-{stable_id(raw['source_id'], final['question'])}",
        "source_id": raw["source_id"], "method": "evol_instruct",
        "backend": raw.get("backend"), "model": raw.get("model"),
        "evolution_depth": len(stages_out), "strategies": raw["strategies"],
        "stages": stages_out,
        **{key: final[key] for key in ("question", "options", "answer", "explanation")},
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evol-Instruct medical MCQ generator")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--depth", type=int, default=3, help="maximum depth; records cycle through 1..depth")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("mlx", "mock"), default="mlx")
    parser.add_argument("--similarity-threshold", type=float, default=0.98)
    parser.add_argument("--max-tokens", type=int, default=2560)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.depth not in (1, 2, 3):
        parser.error("--depth must be 1, 2, or 3")

    sources = read_jsonl(args.input)
    if args.limit > 0:
        sources = sources[: args.limit]
    normalized_sources = []
    for source in sources:
        normalized, error = normalize_mcq(source)
        if error:
            raise SystemExit(f"invalid source {source.get('synthetic_id')}: {error}")
        normalized["synthetic_id"] = source.get("synthetic_id", f"source-{stable_id(normalized['question'])}")
        normalized_sources.append(normalized)

    accepted_path, raw_path, stats_path = output_paths(args.output)
    if raw_path.exists() and not args.resume:
        raw_path.unlink()
    raw_rows = read_jsonl(raw_path) if raw_path.exists() else []
    raw_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        raw_by_source.setdefault(str(row["source_id"]), []).append(row)
    generator = None if args.backend == "mock" else MLXGenerator(args.model, args.max_tokens, args.temperature)
    strategy_names = sorted(EVOL_PROMPTS)

    for index, source in enumerate(normalized_sources):
        source_id = source["synthetic_id"]
        target_depth = 1 + (index % args.depth)
        rng = random.Random(f"{args.seed}:{source_id}")
        strategies = rng.sample(strategy_names, target_depth)
        attempts = raw_by_source.get(source_id, [])
        good = None
        for previous_attempt in attempts:
            good, _ = accepted_record(previous_attempt, source, args.similarity_threshold)
            if good is not None:
                break
        for attempt in range(len(attempts), args.max_retries):
            if good is not None:
                break
            if args.backend == "mock":
                stages = mock_stages(source, strategies)
                response, elapsed = json.dumps({"stages": stages}, ensure_ascii=False), 0.0
            else:
                rng_seed = int(stable_id(args.seed, source_id, attempt, length=8), 16)
                response, elapsed = generator(SYSTEM, build_prompt(source, strategies), rng_seed)
                stages = parse_json_collection(response)
            rng_seed = int(stable_id(args.seed, source_id, attempt, length=8), 16)
            raw = {
                "source_id": source_id, "backend": args.backend, "model": args.model,
                "target_depth": target_depth, "strategies": strategies, "attempt": attempt,
                "rng_seed": rng_seed,
                "response": response, "parsed": stages, "elapsed_s": round(elapsed, 3),
            }
            append_jsonl(raw_path, raw)
            raw_rows.append(raw)
            raw_by_source.setdefault(source_id, []).append(raw)
            good, error = accepted_record(raw, source, args.similarity_threshold)
            if error:
                print(f"[evol] retry source={index} attempt={attempt + 1}/{args.max_retries}: {error}", flush=True)
        if (index + 1) % 25 == 0 or index + 1 == len(normalized_sources):
            print(f"[evol] generated {index + 1}/{len(normalized_sources)}", flush=True)

    accepted = []
    rejects: dict[str, int] = {}
    source_by_id = {row["synthetic_id"]: row for row in normalized_sources}
    for source_id, attempts in raw_by_source.items():
        source = source_by_id.get(source_id)
        if source is None:
            continue
        selected = None
        for raw in attempts:
            record, error = accepted_record(raw, source, args.similarity_threshold)
            if record is not None:
                selected = record
                break
            rejects[error] = rejects.get(error, 0) + 1
        if selected is not None:
            accepted.append(selected)

    write_jsonl(accepted_path, accepted)
    stats = {
        "backend": args.backend, "model": args.model, "source_count": len(normalized_sources),
        "accepted": len(accepted), "acceptance_rate": round(len(accepted) / max(len(normalized_sources), 1), 6),
        "reject_reasons": rejects, "complete": len(accepted) == len(normalized_sources),
        "mlx_rng_seeded": all("rng_seed" in row for row in raw_rows),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evol] wrote {len(accepted)}/{len(normalized_sources)} records -> {accepted_path}")
    if len(accepted) != len(normalized_sources):
        raise SystemExit("some records failed evolution; inspect stats and rerun")


if __name__ == "__main__":
    main()
