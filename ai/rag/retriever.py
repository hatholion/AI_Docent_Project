"""전체 Chroma collection에서 vector similarity로 검색하고 결과를 정규화한다."""

from __future__ import annotations

from typing import Any


def retrieve(store: Any, query: str, config: dict, *, top_k: int | None = None) -> list[dict]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("검색 query는 비어 있지 않은 문자열이어야 합니다")
    count = config["top_k"] if top_k is None else top_k
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("retrieval top_k는 양수여야 합니다")

    # filter를 전달하지 않는다. 현재 유물뿐 아니라 전체 collection이 검색 대상이다.
    pairs = store.similarity_search_with_score(query.strip(), k=count)
    results = []
    for document, distance in pairs:
        chunk_id = document.metadata.get("chunk_id") or document.id
        if not isinstance(chunk_id, str) or not chunk_id:
            raise RuntimeError("Chroma 검색 결과에 chunk_id가 없습니다")
        if document.id is not None and document.id != chunk_id:
            raise RuntimeError("Chroma document ID와 chunk_id metadata가 일치하지 않습니다")
        numeric_distance = float(distance)
        label = document.metadata.get("relic_label") or None
        results.append({
            "chunk_id": chunk_id,
            "relic_label": label,
            "content": document.page_content,
            "metadata": dict(document.metadata),
            "score": 1.0 - numeric_distance,
            "distance": numeric_distance,
        })
    return results
