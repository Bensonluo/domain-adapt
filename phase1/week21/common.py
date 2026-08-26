"""Shared, dependency-light helpers for the Week 21 synthetic-data pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

LETTERS = "ABCDE"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def stable_id(*parts: Any, length: int = 16) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def mcq_content_sha256(row: dict[str, Any]) -> str:
    """Hash every medically material MCQ field for review binding."""
    material = {key: row[key] for key in ("question", "options", "answer", "explanation")}
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_options(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                key = item.get("key", LETTERS[index] if index < len(LETTERS) else "")
                text = item.get("value", item.get("text", ""))
            else:
                key = LETTERS[index] if index < len(LETTERS) else ""
                text = item
            items.append((key, text))
    else:
        return []
    out = []
    for key, text in items:
        letter = str(key).strip().upper()
        cleaned = str(text).strip()
        if letter in LETTERS and cleaned:
            out.append({"key": letter, "value": cleaned})
    out.sort(key=lambda item: LETTERS.index(item["key"]))
    return out


def normalize_mcq(value: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    question = str(value.get("question", value.get("Question", ""))).strip()
    options = normalize_options(value.get("options", value.get("Options")))
    answer = str(value.get("answer", value.get("Answer", ""))).strip().upper()
    explanation = str(value.get("explanation", value.get("Explanation", ""))).strip()
    if len(question) < 8:
        return None, "question_too_short"
    if len(options) not in (4, 5):
        return None, "option_count"
    keys = [option["key"] for option in options]
    if keys != list(LETTERS[: len(options)]):
        return None, "option_keys"
    if answer not in keys:
        return None, "invalid_answer"
    if len(explanation) < 8:
        return None, "explanation_too_short"
    if len(set(option["value"] for option in options)) != len(options):
        return None, "duplicate_options"
    return {
        "question": question,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }, None


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_collection(text: str) -> list[dict[str, Any]]:
    cleaned = strip_code_fence(text)
    # Qwen occasionally closes an explanation with a Chinese opening quote or
    # emits one extra quote/brace between otherwise valid array items.  These
    # repairs are deliberately limited to JSON-boundary positions observed in
    # raw output; question and explanation content is never rewritten.
    repaired = re.sub(r"“(?=\s*[,}\]])", '"', cleaned)
    repaired = repaired.replace('"}},{"question"', '"},{"question"')
    repaired = repaired.replace('"},"{"question"', '"},{"question"')
    repaired = repaired.replace('"},"question"', '"},{"question"')
    candidates = [cleaned, repaired]
    for source in (cleaned, repaired):
        left, right = source.find("["), source.rfind("]")
        if 0 <= left < right:
            candidates.append(source[left : right + 1])
        left, right = source.find("{"), source.rfind("}")
        if 0 <= left < right:
            candidates.append(source[left : right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            for key in ("items", "questions", "data", "stages"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
            else:
                value = [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def compact_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower())


def ngrams(text: str, n: int = 3) -> set[str]:
    value = compact_text(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def max_similarity(question: str, corpus_grams: list[set[str]]) -> float:
    own = ngrams(question)
    return max((jaccard(own, other) for other in corpus_grams), default=0.0)


def prompt_from_mcq(row: dict[str, Any]) -> str:
    options = "\n".join(f"{item['key']}. {item['value']}" for item in row["options"])
    return f"{row['question']}\n{options}\n答案："


class MLXGenerator:
    """Load one local MLX model and expose deterministic text generation."""

    def __init__(self, model_path: str, max_tokens: int, temperature: float) -> None:
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        started = time.time()
        self._mlx_lm = mlx_lm
        self.model, self.tokenizer = mlx_lm.load(model_path)
        self.sampler = make_sampler(temp=temperature)
        self.max_tokens = max_tokens
        print(f"[generator] loaded {model_path} in {time.time() - started:.1f}s", flush=True)

    def __call__(self, system: str, user: str, seed: int) -> tuple[str, float]:
        import mlx.core as mx

        mx.random.seed(seed)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        started = time.time()
        response = self._mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=self.sampler,
            verbose=False,
        )
        return response.strip(), time.time() - started
