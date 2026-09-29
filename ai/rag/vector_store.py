"""기존 Gold chunk_id를 Chroma ID로 사용하는 영속 Vector DB."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _client(config: dict) -> chromadb.ClientAPI:
    path = Path(config["persist_directory"])
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_names(client: chromadb.ClientAPI) -> set[str]:
    return {collection.name for collection in client.list_collections()}


def close_chroma(store: Chroma) -> None:
    """특히 Windows에서 영속 파일 핸들을 명시적으로 반환한다."""
    close = getattr(store._client, "close", None)
    if callable(close):
        close()


def open_chroma(config: dict, embeddings: Embeddings, *, require_existing: bool = True) -> Chroma:
    client = _client(config)
    collection_name = config["collection_name"]
    if require_existing and collection_name not in _collection_names(client):
        raise FileNotFoundError(
            f"Chroma collection이 없습니다: {collection_name}. build_vector_db.py를 먼저 실행하세요"
        )
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": config["embedding_model"],
        },
        create_collection_if_not_exists=not require_existing,
    )


def _batches(documents: list[Document], size: int) -> Iterable[list[Document]]:
    for start in range(0, len(documents), size):
        yield documents[start:start + size]


def build_chroma(documents: list[Document], embeddings: Embeddings, config: dict,
                 *, rebuild: bool = False) -> tuple[Chroma, dict]:
    if not documents:
        raise ValueError("Chroma에 저장할 Gold document가 없습니다")
    ids = [document.metadata.get("chunk_id") for document in documents]
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in ids):
        raise ValueError("모든 document에 chunk_id metadata가 필요합니다")
    if len(ids) != len(set(ids)):
        raise ValueError("중복된 chunk_id가 있습니다")
    if any(document.id != document.metadata["chunk_id"] for document in documents):
        raise ValueError("Document.id와 Gold chunk_id가 일치해야 합니다")
    if any(not document.page_content.strip() for document in documents):
        raise ValueError("비어 있는 embedding document가 있습니다")

    client = _client(config)
    collection_name = config["collection_name"]
    if rebuild and collection_name in _collection_names(client):
        client.delete_collection(collection_name)
    store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": config["embedding_model"],
        },
    )
    current_model = (store._collection.metadata or {}).get("embedding_model")
    if current_model != config["embedding_model"]:
        raise ValueError(
            f"Chroma embedding model 불일치: DB={current_model!r}, config={config['embedding_model']!r}. "
            "--rebuild로 다시 구축하세요"
        )
    for batch in _batches(documents, config["add_batch_size"]):
        store.add_documents(batch, ids=[document.metadata["chunk_id"] for document in batch])

    raw = store._collection.get(include=["documents", "metadatas"])
    stored_ids = raw["ids"]
    if len(stored_ids) != len(documents) or set(stored_ids) != set(ids):
        raise RuntimeError(
            f"Gold/Chroma 개수 또는 ID 불일치: Gold={len(documents)}, Chroma={len(stored_ids)}"
        )
    if any(not text for text in raw["documents"]):
        raise RuntimeError("Chroma에 비어 있는 document가 저장되었습니다")
    if any(not metadata or not metadata.get("chunk_id") for metadata in raw["metadatas"]):
        raise RuntimeError("Chroma metadata에 chunk_id가 누락되었습니다")
    return store, {
        "gold_documents": len(documents),
        "chroma_documents": len(stored_ids),
        "unique_chunk_ids": len(set(stored_ids)),
        "collection_name": collection_name,
        "persist_directory": str(Path(config["persist_directory"])),
    }
