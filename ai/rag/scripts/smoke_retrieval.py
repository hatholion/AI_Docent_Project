"""실제 Gold/Chroma에서 대표 검색을 실행해 ID, label, score를 출력한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.rag.config import load_config, project_path
from ai.rag.embeddings import get_embedding
from ai.rag.retriever import retrieve
from ai.rag.vector_store import open_chroma

DEFAULT_QUERIES = [
    ("specific_relic", "농경문 청동기의 재질과 용도는 무엇인가요?"),
    ("period", "통일신라 시대의 대표적인 석조 불상을 알려줘."),
    ("material", "화강암으로 만든 불교 유물은 무엇인가요?"),
    ("find_place", "서울 암사동 집터유적에서 출토된 유물은?"),
    ("feature", "밭을 일구는 사람과 나무 위 새가 새겨진 유물은?"),
    ("similar", "백자 달항아리와 비슷한 다른 유물을 알려줘."),
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ai/llm/configs/base.yaml")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--query", action="append", help="지정하면 기본 6개 대신 이 질문만 검색")
    args = parser.parse_args()
    config = load_config(args.config)
    vector = dict(config["vector_db"])
    vector["persist_directory"] = str(project_path(vector["persist_directory"]))
    vector["embedding_model"] = config["embedding"]["model"]
    store = open_chroma(vector, get_embedding(config["embedding"]))
    queries = [("custom", value) for value in args.query] if args.query else DEFAULT_QUERIES
    output = []
    for category, query in queries:
        results = retrieve(store, query, config["retrieval"], top_k=args.top_k)
        output.append({
            "category": category,
            "query": query,
            "top_k": args.top_k or config["retrieval"]["top_k"],
            "results": [
                {
                    "chunk_id": result["chunk_id"],
                    "relic_label": result["relic_label"],
                    "score": result["score"],
                    "distance": result["distance"],
                }
                for result in results
            ],
        })
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
