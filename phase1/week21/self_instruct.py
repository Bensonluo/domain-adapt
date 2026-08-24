"""Offline, resumable Self-Instruct generation for Chinese medical MCQs.

The production backend uses a local MLX model.  ``--backend mock`` exists only
for deterministic tests and pipeline smoke runs; output records retain the
backend name so mock data cannot be mistaken for experimental evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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

SYSTEM = (
    "你是严谨的中国医学考试命题专家。只输出合法 JSON，不输出 Markdown、思考过程或额外文字。"
    "每题必须是原创单项选择题，知识正确，只有一个最佳答案。"
)


def output_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.suffix == ".jsonl":
        return output, output.with_name("self_instruct_raw.jsonl"), output.with_name("self_instruct_stats.json")
    return output / "self_instruct.jsonl", output / "self_instruct_raw.jsonl", output / "self_instruct_stats.json"


def render_examples(examples: list[dict[str, Any]]) -> str:
    compact = []
    for row in examples:
        compact.append({
            "question": row["question"], "options": row["options"],
            "answer": row["answer"], "explanation": row["explanation"],
        })
    return json.dumps(compact, ensure_ascii=False)


def generation_prompt(examples: list[dict[str, Any]], count: int, generation_id: int) -> str:
    return f"""参考下面的医学单选题风格，但不要复制题干、病例、数值或选项：
{render_examples(examples)}

生成 {count} 道彼此不同的新题。覆盖诊断、治疗、药理、基础医学或公共卫生中的不同知识点；优先采用临床情境。
严格返回 JSON 数组，每个对象恰含：
- question: 至少 8 个字的中文题干
- options: 4 或 5 个对象，每个对象含 key(A-E连续) 和 value
- answer: 唯一正确选项字母
- explanation: 至少一句，解释正确项并说明关键鉴别点
批次编号：{generation_id}。只返回 JSON 数组。"""


def mock_candidates(examples: list[dict[str, Any]], count: int, generation_id: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        source = examples[index % len(examples)]
        options = [dict(option) for option in source["options"]]
        rows.append({
            "question": f"模拟批次{generation_id}-{index}：在新的临床情境中，{source['question']}",
            "options": options,
            "answer": source["answer"],
            "explanation": f"模拟数据仅用于测试。原始知识点的正确选项为 {source['answer']}。",
        })
    return rows


def replay_raw(raw_rows: list[dict[str, Any]], seeds: list[dict[str, Any]], threshold: float, target: int):
    accepted: list[dict[str, Any]] = []
    corpus = [ngrams(seed["question"]) for seed in seeds]
    rejects: Counter[str] = Counter()
    seen_questions: set[str] = set()
    for raw in sorted(raw_rows, key=lambda row: int(row["generation_id"])):
        values = raw.get("parsed")
        if not isinstance(values, list) or not values:
            values = parse_json_collection(str(raw.get("response", "")))
        for index, value in enumerate(values):
            normalized, error = normalize_mcq(value)
            if error:
                rejects[error] += 1
                continue
            question = normalized["question"]
            if question in seen_questions:
                rejects["exact_duplicate"] += 1
                continue
            similarity = max_similarity(question, corpus)
            if similarity >= threshold:
                rejects["too_similar"] += 1
                continue
            record = {
                "synthetic_id": f"self-{stable_id(raw['generation_id'], index, question)}",
                **normalized,
                "method": "self_instruct",
                "backend": raw.get("backend"),
                "model": raw.get("model"),
                "generation_id": int(raw["generation_id"]),
                "seed_ids": raw.get("seed_ids", []),
                "max_similarity_at_accept": round(similarity, 6),
            }
            accepted.append(record)
            seen_questions.add(question)
            corpus.append(ngrams(question))
            if len(accepted) >= target:
                return accepted, rejects
    return accepted, rejects


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Instruct medical MCQ generator")
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--model", required=True, help="local MLX model path or label for mock")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("mlx", "mock"), default="mlx")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-attempt-multiplier", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.batch_size <= 0:
        parser.error("--n and --batch-size must be positive")
    if not 0.0 < args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be in (0, 1]")

    seeds = []
    for raw in read_jsonl(args.seeds):
        normalized, error = normalize_mcq(raw)
        if error:
            raise SystemExit(f"invalid seed {raw.get('seed_id')}: {error}")
        normalized["seed_id"] = raw.get("seed_id", f"seed-{stable_id(normalized['question'])}")
        seeds.append(normalized)
    if len(seeds) < 3:
        raise SystemExit("at least 3 valid seeds are required")

    accepted_path, raw_path, stats_path = output_paths(args.output)
    if raw_path.exists() and not args.resume:
        raw_path.unlink()
    raw_rows = read_jsonl(raw_path) if raw_path.exists() else []
    accepted, rejects = replay_raw(raw_rows, seeds, args.similarity_threshold, args.n)
    done_ids = {int(row["generation_id"]) for row in raw_rows}
    next_id = max(done_ids, default=-1) + 1
    max_attempts = max(1, (args.n + args.batch_size - 1) // args.batch_size) * args.max_attempt_multiplier
    generator = None if args.backend == "mock" else MLXGenerator(args.model, args.max_tokens, args.temperature)

    for generation_id in range(next_id, max_attempts):
        if len(accepted) >= args.n:
            break
        rng = random.Random(f"{args.seed}:{generation_id}")
        example_count = min(len(seeds), rng.randint(3, 5))
        examples = rng.sample(seeds, example_count)
        request_count = min(args.batch_size, args.n - len(accepted) + 1)
        if args.backend == "mock":
            parsed = mock_candidates(examples, request_count, generation_id)
            response, elapsed = json.dumps(parsed, ensure_ascii=False), 0.0
        else:
            response, elapsed = generator(
                SYSTEM, generation_prompt(examples, request_count, generation_id), args.seed + generation_id
            )
            parsed = parse_json_collection(response)
        raw = {
            "generation_id": generation_id,
            "backend": args.backend,
            "model": args.model,
            "seed_ids": [row["seed_id"] for row in examples],
            "rng_seed": args.seed + generation_id,
            "response": response,
            "parsed": parsed,
            "elapsed_s": round(elapsed, 3),
        }
        append_jsonl(raw_path, raw)
        raw_rows.append(raw)
        accepted, rejects = replay_raw(raw_rows, seeds, args.similarity_threshold, args.n)
        if (generation_id + 1) % 10 == 0 or len(accepted) >= args.n:
            print(f"[self] generation={generation_id + 1}/{max_attempts} accepted={len(accepted)}/{args.n}", flush=True)

    write_jsonl(accepted_path, accepted)
    stats = {
        "backend": args.backend, "model": args.model, "target": args.n,
        "accepted": len(accepted), "raw_generations": len(raw_rows),
        "acceptance_rate": round(len(accepted) / max(sum(len(row.get('parsed', [])) for row in raw_rows), 1), 6),
        "similarity_threshold": args.similarity_threshold, "reject_reasons": dict(rejects),
        "mlx_rng_seeded": all("rng_seed" in row for row in raw_rows),
        "complete": len(accepted) == args.n,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[self] wrote {len(accepted)} accepted records -> {accepted_path}")
    if len(accepted) < args.n:
        raise SystemExit(f"generation exhausted: accepted {len(accepted)} of {args.n}")


if __name__ == "__main__":
    main()
