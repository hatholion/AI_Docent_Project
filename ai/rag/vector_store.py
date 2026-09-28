"""실험별 독립 Chroma 저장소."""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def create_chroma(chunks: list[Document], embeddings: Embeddings, config: dict,
                  experiment_name: str, rebuild: bool = False) -> Chroma:
    base = Path(config["base_dir"]) / experiment_name / "chroma"
    if base.exists() and any(base.iterdir()):
        if not rebuild:
            raise FileExistsError(f"기존 DB가 있습니다: {base} (--rebuild 필요)")
        import shutil
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    store = Chroma(collection_name=config["collection_name"],
                   embedding_function=embeddings, persist_directory=str(base))
    ids = [doc.metadata["chunk_id"] for doc in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("중복된 chunk_id가 있습니다")
    store.add_documents(chunks, ids=ids)
    return store
