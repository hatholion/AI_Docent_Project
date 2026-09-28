"""교체 가능한 Gold 문서 청킹."""

import re
from collections import Counter
from hashlib import sha256
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ai.rag.document_builder import field_sections


def _sentence_chunks(text: str, size: int, overlap: int) -> list[str]:
    # 한국어 종결부호와 줄바꿈을 문장 경계로 취급한다.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current = (current + " " + sentence).strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: list[Document], config: dict,
                    gold_rows: list[dict] | None = None) -> list[Document]:
    """내용 기반 안정 ID와 relic_id, chunk_index, chunk_type을 보존한다."""
    strategy = config["strategy"]
    size, overlap = config["chunk_size"], config["chunk_overlap"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    rows_by_id = {r["relic_id"]: r for r in gold_rows or []}
    output = []
    for doc in documents:
        if strategy == "none":
            pieces = [("full", doc.page_content)]
        elif strategy == "recursive":
            pieces = [("recursive", x) for x in splitter.split_text(doc.page_content)]
        elif strategy == "sentence_based":
            pieces = [("sentence", x) for x in _sentence_chunks(doc.page_content, size, overlap)]
        elif strategy == "field_based":
            row = rows_by_id.get(doc.metadata["relic_id"])
            if row is None:
                raise ValueError("field_based에는 gold_rows가 필요합니다")
            pieces = [(group, text) for group, text in field_sections(row).items() if text]
        else:
            raise ValueError(f"지원하지 않는 청킹 방식: {strategy}")
        occurrences: Counter[str] = Counter()
        for index, (kind, content) in enumerate(pieces):
            digest = sha256(content.encode("utf-8")).hexdigest()
            base_id = f"{doc.metadata['relic_id']}:{strategy}:{kind}:{digest}"
            occurrences[base_id] += 1
            chunk_id = base_id if occurrences[base_id] == 1 else f"{base_id}:{occurrences[base_id]}"
            output.append(Document(id=chunk_id, page_content=content,
                                   metadata={**doc.metadata, "chunk_index": index,
                                             "chunk_type": kind, "chunk_id": chunk_id}))
    return output
