"""Gold chunk_id를 정답으로 사용하는 Retriever 평가."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ai.rag.retriever import retrieve

KS = (1, 3, 5)


def load_evaluation(path: str | Path) -> list[dict]:
    source = Path(path)
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("evaluation dataset은 비어 있지 않은 JSON 배열이어야 합니다")
    seen_ids: set[str] = set()
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"평가 {number}번: JSON 객체가 필요합니다")
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
            raise ValueError(f"평가 {number}번: 고유한 문자열 id가 필요합니다")
        seen_ids.add(sample_id)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"평가 {number}번: question이 필요합니다")
        relevant = item.get("relevant_chunk_ids")
        if (not isinstance(relevant, list) or not relevant
                or any(not isinstance(value, str) or not value for value in relevant)
                or len(set(relevant)) != len(relevant)):
            raise ValueError(f"평가 {number}번: relevant_chunk_ids는 중복 없는 문자열 배열이어야 합니다")
        labels = item.get("relevant_relic_labels")
        if not isinstance(labels, list) or any(not isinstance(value, str) or not value for value in labels):
            raise ValueError(f"평가 {number}번: relevant_relic_labels는 문자열 배열이어야 합니다")
        if "retrieval_query" in item and (
            not isinstance(item["retrieval_query"], str) or not item["retrieval_query"].strip()
        ):
            raise ValueError(f"평가 {number}번: retrieval_query는 비어 있지 않은 문자열이어야 합니다")
    return items


def validate_answers(items: list[dict], valid_chunk_ids: set[str]) -> None:
    for item in items:
        unknown = set(item["relevant_chunk_ids"]) - valid_chunk_ids
        if unknown:
            raise ValueError(f"평가 {item['id']}: Gold/Chroma에 없는 relevant_chunk_ids: {sorted(unknown)}")


def validate_no_relic_labels_in_queries(items: list[dict], relic_labels: set[str]) -> None:
    """사용자에게 노출되지 않는 소장품 번호가 검색 질의에 섞이는 것을 막는다."""
    labels = sorted((label for label in relic_labels if label), key=len, reverse=True)
    for item in items:
        for field in ("question", "retrieval_query"):
            text = item.get(field)
            if not text:
                continue
            matched = next((label for label in labels if label in text), None)
            if matched:
                raise ValueError(
                    f"평가 {item['id']}: {field}에 사용자에게 주어지지 않는 "
                    f"relic_label '{matched}'이 포함되어 있습니다"
                )


def evaluate(store, dataset: list[dict], retrieval_config: dict, *,
             top_k: int | None = None,
             retrieve_fn: Callable = retrieve) -> dict:
    requested_k = retrieval_config["top_k"] if top_k is None else top_k
    if isinstance(requested_k, bool) or not isinstance(requested_k, int) or requested_k <= 0:
        raise ValueError("top_k는 양수여야 합니다")
    search_k = max(max(KS), requested_k)
    totals = {f"recall_at_{k}": 0.0 for k in KS}
    reciprocal_rank_total = 0.0
    details = []
    for item in dataset:
        query = item.get("retrieval_query") or item["question"]
        results = retrieve_fn(store, query, retrieval_config, top_k=search_k)
        retrieved_ids = [result["chunk_id"] for result in results]
        relevant = set(item["relevant_chunk_ids"])
        metrics = {
            f"recall_at_{k}": len(relevant.intersection(retrieved_ids[:k])) / len(relevant)
            for k in KS
        }
        first_rank = next((rank for rank, chunk_id in enumerate(retrieved_ids, 1)
                           if chunk_id in relevant), None)
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        reciprocal_rank_total += reciprocal_rank
        for key, value in metrics.items():
            totals[key] += value
        details.append({
            "id": item["id"],
            "category": item.get("category"),
            "question": item["question"],
            "retrieval_query": query,
            "current_relic_label": item.get("current_relic_label"),
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            "relevant_relic_labels": item["relevant_relic_labels"],
            "retrieved_chunk_ids": retrieved_ids,
            "retrieved_relic_labels": [result["relic_label"] for result in results],
            "scores": [result["score"] for result in results],
            "distances": [result["distance"] for result in results],
            **metrics,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": reciprocal_rank,
        })
    count = len(dataset)
    return {
        "total_queries": count,
        "top_k": search_k,
        "recall_at_1": totals["recall_at_1"] / count,
        "recall_at_3": totals["recall_at_3"] / count,
        "recall_at_5": totals["recall_at_5"] / count,
        "mrr": reciprocal_rank_total / count,
        "details": details,
    }
