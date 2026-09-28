"""선택된 유물의 청크만 검색하는 Retriever."""

from langchain_core.documents import Document


def require_relic_id(relic_id: str | None) -> str:
    """검색 범위를 지정하지 않은 전체 DB 검색을 차단한다."""
    if not isinstance(relic_id, str) or not relic_id.strip():
        raise ValueError("relic_id는 필수입니다. 선택한 relic_label을 내부 ID로 변환한 뒤 전달하세요")
    return relic_id.strip()


def retrieve(store, query: str, config: dict, *, relic_id: str | None = None,
             k: int | None = None) -> list[Document]:
    """relic_id metadata 필터 안에서 유사도 또는 MMR 순위를 매긴다."""
    selected_id = require_relic_id(relic_id)
    count = config["k"] if k is None else k
    if count <= 0:
        raise ValueError("retriever k는 양수여야 합니다")
    filter_by_relic = {"relic_id": selected_id}
    if config["search_type"] == "similarity":
        docs = store.similarity_search(query, k=count, filter=filter_by_relic)
    elif config["search_type"] == "mmr":
        docs = store.max_marginal_relevance_search(
            query, k=count, fetch_k=max(config["fetch_k"], count),
            lambda_mult=config["lambda_mult"], filter=filter_by_relic,
        )
    else:
        raise ValueError("지원하지 않는 검색 방식")
    if any(doc.metadata.get("relic_id") != selected_id for doc in docs):
        raise RuntimeError("Chroma 필터 결과에 다른 relic_id가 포함되었습니다")
    return docs
