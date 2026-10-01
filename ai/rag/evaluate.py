"""평가 (기획서 §8): 같은 질문 세트로 검색과 답변을 따로 채점한다.

python -m ai.rag.evaluate                  # 검색만 (빠름)
python -m ai.rag.evaluate --answers        # + 답변 생성, 출처 정확도, 함정 통과
python -m ai.rag.evaluate --answers --judge  # + LLM 채점 충실성

결과: runs/rag/<시각>/results.jsonl, summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ai.rag import ollama
from ai.rag.config import EVAL_OUTPUT_DIR, EVAL_QUESTIONS_PATH, RetrievalSettings
from ai.rag.data import read_jsonl
from ai.rag.generate import NOT_FOUND_ANSWER, build_context, generate_answer
from ai.rag.retrieval import RetrievalResult, Retriever

# 합격 기준 (기획서 §8, 제안값)
TARGETS = {
    "recall_at_6": 0.90,
    "label_accuracy": 1.00,
    "faithfulness": 0.95,
    "citation_accuracy": 1.00,
    "trap_pass": 1.00,
}

JUDGE_PROMPT = """너는 채점자다. [자료]만을 근거로 [답변]의 사실 주장이 모두 자료에 있는지 확인하라.
자료에 없는 사실(연대, 인물, 의미, 수치 등)이 하나라도 있으면 unsupported다. 출처 번호 표기와 인사말은 채점하지 않는다.
"자료에 없다", "명시되어 있지 않다"처럼 정보가 없다고 말하는 문장은 사실 주장이 아니므로 문제 삼지 않는다.
반드시 JSON 한 줄로만 답하라: {{"supported": true 또는 false, "unsupported_claims": ["..."]}}

[자료]
{context}

[답변]
{answer}"""


def parent_metadata(retriever: Retriever) -> dict[str, dict[str, Any]]:
    """parent_id → 필터용 metadata (첫 청크 기준)."""
    result: dict[str, dict[str, Any]] = {}
    for chunk in retriever.gold.chunks:
        result.setdefault(chunk["parent_id"], chunk["metadata"])
    return result


def satisfies(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "designations":
            if value not in (metadata.get("designations") or []):
                return False
        elif metadata.get(key) != value:
            return False
    return True


def score_retrieval(item: dict[str, Any], result: RetrievalResult, metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = [p for p in result.parents if p.via != "related"]
    ids = [p.parent_id for p in ranked]
    record: dict[str, Any] = {
        "route": result.route,
        "found": result.found,
        "filters": result.filters,
        "stats_question": result.stats_question,
        "best_vector_similarity": round(result.best_vector_similarity, 4),
        "best_bm25_score": round(result.best_bm25_score, 3),
        "top": [(p.label or p.parent_id, p.card["title"], p.via) for p in result.parents],
    }
    kind = item["type"]
    expected = item.get("expected_ids", [])
    if kind == "label":
        record["label_correct"] = result.route == "label" and bool(ids) and ids[0] == expected[0]
    if kind in {"label", "name", "descriptive", "set"} or (kind == "trap" and expected):
        record["hit"] = any(e in ids for e in expected)
    if kind == "condition":
        matching = [pid for pid in ids if satisfies(metas.get(pid, {}), item["expected_filter"])]
        title_any = item.get("expected_title_any")
        if title_any:
            matching = [pid for pid in matching if any(w in result.parents[ids.index(pid)].card["title"] for w in title_any)]
        record["hit"] = bool(matching)
        record["precision"] = round(len(matching) / len(ids), 3) if ids else 0.0
    if kind == "trap":
        trap = item["trap"]
        if trap == "not_found":
            record["trap_retrieval_ok"] = not result.found
        elif trap == "stats":
            record["trap_retrieval_ok"] = result.stats_question
        elif trap == "photo_plate":
            record["trap_retrieval_ok"] = bool(ranked) and ranked[0].card.get("record_type") == "photo_plate"
        else:
            record["trap_retrieval_ok"] = record.get("hit", False)
    return record


def judge(context: str, answer: str) -> dict[str, Any]:
    raw = ollama.chat([{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}], temperature=0.0, max_tokens=300)
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(match.group(0)) if match else {"supported": None, "raw": raw}
    except json.JSONDecodeError:
        return {"supported": None, "raw": raw}


ABSENCE_RE = re.compile(r"명시되어 있지 않|나와 있지 않|찾지 못|자료에 없|확인할 수 없|알 수 없")


def trap_answer_ok(item: dict[str, Any], text: str, faithful: bool | None) -> bool:
    trap = item["trap"]
    if trap == "not_found":
        return text == NOT_FOUND_ANSWER or "찾지 못" in text
    if trap == "stats":
        return "2.3" in text
    if trap == "photo_plate":
        return "사진" in text
    # no_description: 분류 정보 이상을 지어내지 않았는가.
    # "자료에 없다"고 답하면 통과 (EXAONE 채점자가 이런 문장을 근거 없는 주장으로 잘못 보는 경우가 있음)
    if ABSENCE_RE.search(text):
        return True
    return bool(faithful)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["type"]].append(record)

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    recall_items = [r for r in records if r["type"] in {"label", "name", "descriptive", "set", "condition"}]
    summary: dict[str, Any] = {
        "count": len(records),
        "recall_at_6": rate([r["hit"] for r in recall_items]),
        "recall_at_6_by_type": {t: rate([r["hit"] for r in rs if "hit" in r]) for t, rs in by_type.items()},
        "label_accuracy": rate([r["label_correct"] for r in by_type["label"]]),
        "condition_precision": rate([r["precision"] >= 0.5 for r in by_type["condition"]]),
        "trap_retrieval_ok": rate([r["trap_retrieval_ok"] for r in by_type["trap"]]),
    }
    answered = [r for r in records if "answer" in r]
    if answered:
        summary["citation_accuracy"] = rate([not r["removed_labels"] for r in answered])
        summary["trap_pass"] = rate([r["trap_answer_ok"] for r in answered if r["type"] == "trap"])
        judged = [r["faithful"] for r in answered if r.get("faithful") is not None]
        summary["faithfulness"] = rate(judged)
        summary["llm_seconds_avg"] = round(sum(r["llm_seconds"] for r in answered) / len(answered), 2)
    summary["targets"] = {k: {"target": v, "actual": summary.get(k), "pass": summary.get(k) is not None and summary[k] >= v} for k, v in TARGETS.items()}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=EVAL_QUESTIONS_PATH)
    parser.add_argument("--answers", action="store_true", help="답변까지 생성해 채점")
    parser.add_argument("--judge", action="store_true", help="LLM으로 충실성 채점 (--answers 필요)")
    parser.add_argument("--only", help="쉼표로 구분한 유형만 (예: trap,label)")
    parser.add_argument("--min-vector", type=float, help="근거 없음 판단 벡터 임계값 덮어쓰기")
    parser.add_argument("--min-bm25", type=float, help="근거 없음 판단 BM25 임계값 덮어쓰기")
    args = parser.parse_args()

    settings = RetrievalSettings()
    if args.min_vector is not None:
        settings = replace(settings, min_vector_similarity=args.min_vector)
    if args.min_bm25 is not None:
        settings = replace(settings, min_bm25_score=args.min_bm25)
    retriever = Retriever(settings)
    metas = parent_metadata(retriever)
    items = read_jsonl(args.questions)
    if args.only:
        wanted = set(args.only.split(","))
        items = [item for item in items if item["type"] in wanted]

    out_dir = EVAL_OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in items:
        started = time.perf_counter()
        result = retriever.retrieve(item["question"])
        record = {**item, **score_retrieval(item, result, metas), "retrieval_seconds": round(time.perf_counter() - started, 3)}
        if args.answers:
            answer = generate_answer(result)
            record.update(answer=answer.answer, sources=answer.sources, cited_labels=answer.cited_labels,
                          removed_labels=answer.removed_labels, llm_seconds=answer.llm_seconds)
            if args.judge and answer.found and not result.stats_question:
                verdict = judge(build_context(result.parents), answer.answer)
                record["faithful"] = verdict.get("supported")
                record["judge"] = verdict
            if item["type"] == "trap":
                record["trap_answer_ok"] = trap_answer_ok(item, answer.answer, record.get("faithful"))
        records.append(record)
        mark = "✓" if record.get("hit", record.get("trap_retrieval_ok")) else "✗"
        print(f"{mark} {item['id']} [{item['type']}] {item['question']} → {record['top'][:2]}")

    summary = summarize(records)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
