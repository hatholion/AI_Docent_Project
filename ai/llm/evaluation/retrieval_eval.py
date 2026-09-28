"""선택 유물 안의 관련 청크 검색률을 평가한다."""

from __future__ import annotations

import json
from pathlib import Path

from ai.rag.retriever import retrieve
from ai.rag.gold_loader import require_relic_label, resolve_relic_id

KS = (1, 3, 5)


def _validate_item(item: dict, number: int) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip():
        raise ValueError(f"평가 {number}번: 비어 있지 않은 query가 필요합니다")
    try:
        if "relic_id" in item:
            raise ValueError("평가 입력의 relic_id 대신 relic_label에 소장품번호를 지정하세요")
        if not isinstance(item.get("relic_label"), str) or not item["relic_label"].strip():
            raise ValueError("평가 JSON 항목의 relic_label에 소장품번호를 지정하세요")
        require_relic_label(item.get("relic_label"))
    except ValueError as exc:
        raise ValueError(f"평가 {number}번: {exc}") from exc
    relevant = item.get("relevant_chunk_ids")
    if (not isinstance(relevant, list) or not relevant
            or any(not isinstance(value, str) or not value for value in relevant)
            or len(set(relevant)) != len(relevant)):
        raise ValueError(f"평가 {number}번: relevant_chunk_ids는 중복 없는 문자열 배열이어야 합니다")


def load_evaluation(path: str | Path) -> list[dict]:
    """사람이 라벨링한 질문·소장품번호·관련 청크 ID를 읽는다."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("evaluation dataset은 배열이어야 합니다")
    for number, item in enumerate(items, 1):
        _validate_item(item, number)
    return items


def evaluate(store, dataset: list[dict], retriever_config: dict, *, gold_rows: list[dict]) -> dict:
    """LLM 없이 각 유물의 Top-1/3/5 청크 Recall을 계산한다."""
    if not dataset:
        raise ValueError("평가 항목이 없습니다")
    selections = []
    for number, item in enumerate(dataset, 1):
        _validate_item(item, number)
        label = require_relic_label(item["relic_label"])
        try:
            selections.append((label, resolve_relic_id(gold_rows, label)))
        except ValueError as exc:
            raise ValueError(f"평가 {number}번: {exc}") from exc
    totals = {f"chunk_recall_at_{k}": 0.0 for k in KS}
    per_query = []
    for item, (label, relic_id) in zip(dataset, selections):
        docs = retrieve(store, item["query"], retriever_config,
                        relic_id=relic_id, k=5)
        retrieved_ids = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id or doc.id != chunk_id:
                raise ValueError("Chroma의 chunk_id metadata/Document.id가 없거나 일치하지 않습니다. 재인덱싱하세요")
            retrieved_ids.append(chunk_id)
        relevant = set(item["relevant_chunk_ids"])
        metrics = {f"chunk_recall_at_{k}": len(relevant.intersection(retrieved_ids[:k])) / len(relevant)
                   for k in KS}
        for key, value in metrics.items():
            totals[key] += value
        per_query.append({"query": item["query"], "relic_label": label, "relic_id": relic_id,
                          "relevant_chunk_ids": item["relevant_chunk_ids"],
                          "retrieved_chunk_ids": retrieved_ids, **metrics})
    return {"per_query": per_query,
            "averages": {key: value / len(dataset) for key, value in totals.items()},
            "counts": {"queries": len(dataset)}}
