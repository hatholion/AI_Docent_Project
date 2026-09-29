"""문서와 질문에 동일한 YAML 기반 embedding model을 적용한다."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class PrefixedEmbeddings(Embeddings):
    def __init__(self, model: HuggingFaceEmbeddings, query_prefix: str, document_prefix: str):
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(self.query_prefix + text)


@lru_cache(maxsize=4)
def _model(model_name: str, device: str, normalize: bool, batch_size: int) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize, "batch_size": batch_size},
        query_encode_kwargs={"normalize_embeddings": normalize},
        show_progress=True,
    )


def get_embedding(config: dict) -> Embeddings:
    device = config["device"]
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _model(config["model"], device, config["normalize_embeddings"], config["batch_size"])
    return PrefixedEmbeddings(model, config.get("query_prefix", ""), config.get("document_prefix", ""))
