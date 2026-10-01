"""Quantitative comparison of local LLM candidates for the docent chatbot.

Golden Set: for each of the 7 artifacts, ask a factual question (material/period)
and a designation question (국보/보물 여부) grounded in that artifact's GOLD
context_text. The designation question doubles as an automatic hallucination
probe: 3 artifacts are actually designated (bon002789, jub000702, ssu001794)
and 4 are not, so we know the ground truth and can flag suspicious answers.

Usage:
    python -m ai.llm.compare_models
    python -m ai.llm.compare_models --models exaone3.5:7.8b llama3.1:latest

Output: ai/llm/comparison_results.json (raw) + a summary table printed to stdout
and written to ai/llm/comparison_summary.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ai.llm.ollama_client import generate
from ai.rag.gold_store import get_context_text
from ai.rag.prompts import build_chat_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum" / "gold_artifacts.jsonl"

DEFAULT_MODELS = ["exaone3.5:7.8b", "llama3.1:latest", "qwen2.5-7b-quant-manual"]

DESIGNATED = {"bon002789", "jub000702", "ssu001794"}
DESIGNATION_KEYWORDS = ["국보", "보물", "천연기념물"]
# "사적"/"명승"은 "역사적"처럼 흔한 단어의 부분 문자열이라 오탐을 유발해서 뺐다.
# 우리 GOLD 데이터의 지정문화재도 전부 국보/보물뿐이라 실질적으로 문제 없음.
NEGATION_WORDS = ["아니", "않", "없", "미지정", "못"]

QUESTION_TEMPLATES = [
    ("factual", "이 유물은 언제(시대), 무엇으로(재질) 만들어졌어?"),
    ("designation", "이 유물이 국보나 보물로 지정되어 있어? 지정 안 됐으면 안 됐다고 말해줘."),
]


def load_gold_records() -> list[dict]:
    with GOLD_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_golden_set(records: list[dict]) -> list[dict]:
    golden_set = []
    for record in records:
        for qtype, question in QUESTION_TEMPLATES:
            golden_set.append(
                {
                    "artifact_id": record["artifact_id"],
                    "artifact_name": record["artifact_name"],
                    "question_type": qtype,
                    "question": question,
                    "material_l1": record.get("material_l1"),
                    "material_l2": record.get("material_l2"),
                    "period": record.get("period"),
                    "is_designated": record["artifact_id"] in DESIGNATED,
                }
            )
    return golden_set


def split_sentences(text: str) -> list[str]:
    import re

    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def check_designation_flag(response: str, is_designated: bool) -> bool:
    """True면 '검토 필요' 표시. 지정 안 된 유물인데, 지정 키워드가 나온 문장에
    부정어가 하나도 없으면 의심(같은 문장 기준이라 문장 길이에 안 흔들림)."""
    if is_designated:
        return False
    for sentence in split_sentences(response):
        if any(keyword in sentence for keyword in DESIGNATION_KEYWORDS):
            if not any(neg in sentence for neg in NEGATION_WORDS):
                return True
    return False


def check_material_mentioned(response: str, material_l1: str | None, material_l2: str | None) -> bool:
    return bool((material_l1 and material_l1 in response) or (material_l2 and material_l2 in response))


def check_period_mentioned(response: str, period: str | None) -> bool:
    return bool(period and period in response)


def run_model(model: str, golden_set: list[dict], context_by_id: dict[str, str]) -> list[dict]:
    results = []
    for item in golden_set:
        context_text = context_by_id[item["artifact_id"]]
        prompt = build_chat_prompt(context_text, "general", item["question"])
        start = time.time()
        response = generate(prompt, model)
        elapsed = time.time() - start
        text = response.get("response", "").strip()

        result = {
            **item,
            "model": model,
            "response": text,
            "elapsed_sec": round(elapsed, 2),
            "eval_count": response.get("eval_count"),
            "material_mentioned": check_material_mentioned(
                text, item["material_l1"], item["material_l2"]
            ),
            "period_mentioned": check_period_mentioned(text, item["period"]),
            "designation_flag": check_designation_flag(text, item["is_designated"]),
        }
        results.append(result)
        print(
            f"[{model}] {item['artifact_id']}/{item['question_type']}: "
            f"{elapsed:.2f}s flag={result['designation_flag']}"
        )
    return results


def summarize(results: list[dict], models: list[str]) -> str:
    lines = [
        "| 모델 | 평균 응답시간(초) | 평균 토큰수 | 재질 언급률 | 시대 언급률 | 지정문화재 오답 의심 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        rows = [r for r in results if r["model"] == model]
        n = len(rows)
        avg_time = sum(r["elapsed_sec"] for r in rows) / n
        avg_tokens = sum(r["eval_count"] or 0 for r in rows) / n
        material_rate = sum(r["material_mentioned"] for r in rows) / n * 100
        period_rate = sum(r["period_mentioned"] for r in rows) / n * 100
        flags = sum(r["designation_flag"] for r in rows)
        lines.append(
            f"| {model} | {avg_time:.2f} | {avg_tokens:.0f} | "
            f"{material_rate:.0f}% | {period_rate:.0f}% | {flags}건 |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "comparison_results.json"
    )
    parser.add_argument(
        "--summary", type=Path, default=Path(__file__).resolve().parent / "comparison_summary.md"
    )
    args = parser.parse_args()

    records = load_gold_records()
    golden_set = build_golden_set(records)
    context_by_id = {r["artifact_id"]: r["context_text"] for r in records}

    print(f"golden_set_size={len(golden_set)} models={args.models}")

    all_results: list[dict] = []
    for model in args.models:
        all_results.extend(run_model(model, golden_set, context_by_id))

    args.output.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_md = summarize(all_results, args.models)
    args.summary.write_text(summary_md + "\n", encoding="utf-8")

    print("\n" + summary_md)
    print(f"\nraw_output={args.output}")
    print(f"summary_output={args.summary}")


if __name__ == "__main__":
    main()
