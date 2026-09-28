"""선택된 유물 청크에서 LLM에 전달할 컨텍스트를 만든다."""

from langchain_core.documents import Document
from ai.rag.retriever import retrieve


def build_context(store, question: str, relic_id: str,
                  retriever_config: dict) -> tuple[str, list[Document]]:
    """relic_id 필터 검색과 출처 ID가 붙은 컨텍스트 구성."""
    docs = retrieve(store, question, retriever_config, relic_id=relic_id)
    context = "\n\n".join(
        f"[유물 {doc.metadata['relic_id']}]\n{doc.page_content}" for doc in docs
    )
    return context, docs
