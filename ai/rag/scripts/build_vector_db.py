"""완성된 Gold의 각 레코드를 그대로 Chroma document 하나로 구축한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rag.config import load_config, project_path
from ai.rag.document_builder import build_documents
from ai.rag.embeddings import get_embedding
from ai.rag.gold_loader import gold_summary, load_gold
from ai.rag.vector_store import build_chroma


def run(config_file: str, *, rebuild: bool = False) -> dict:
    config = load_config(config_file)
    gold_path = project_path(config["data"]["gold_path"])
    rows = load_gold(gold_path)
    documents = build_documents(rows)
    embeddings = get_embedding(config["embedding"])
    vector = dict(config["vector_db"])
    vector["persist_directory"] = str(project_path(vector["persist_directory"]))
    vector["embedding_model"] = config["embedding"]["model"]
    _, chroma_stats = build_chroma(documents, embeddings, vector, rebuild=rebuild)
    return {
        "gold_path": str(gold_path),
        **gold_summary(rows),
        **chroma_stats,
        "embedding_model": config["embedding"]["model"],
        "chunking_performed": False,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ai/llm/configs/base.yaml")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, rebuild=args.rebuild), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
