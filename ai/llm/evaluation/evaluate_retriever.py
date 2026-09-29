"""영속 Chroma Retriever의 Recall@1/3/5와 MRR을 계산한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.llm.evaluation.retrieval_eval import (
    evaluate,
    load_evaluation,
    validate_answers,
    validate_no_relic_labels_in_queries,
)
from ai.rag.config import load_config, project_path
from ai.rag.embeddings import get_embedding
from ai.rag.gold_loader import load_gold
from ai.rag.vector_store import open_chroma


def run(config_file: str, dataset_file: str, *, top_k: int | None = None,
        output_file: str | None = None) -> dict:
    config = load_config(config_file)
    rows = load_gold(project_path(config["data"]["gold_path"]))
    dataset = load_evaluation(project_path(dataset_file))
    validate_answers(dataset, {row["chunk_id"] for row in rows})
    validate_no_relic_labels_in_queries(
        dataset,
        {row["relic_label"] for row in rows if row.get("relic_label")},
    )
    vector = dict(config["vector_db"])
    vector["persist_directory"] = str(project_path(vector["persist_directory"]))
    vector["embedding_model"] = config["embedding"]["model"]
    store = open_chroma(vector, get_embedding(config["embedding"]))
    result = evaluate(store, dataset, config["retrieval"], top_k=top_k)
    if output_file:
        output = project_path(output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ai/llm/configs/base.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.config, args.dataset, top_k=args.top_k, output_file=args.output)
    print(f"Total Queries: {result['total_queries']}")
    print(f"Recall@1: {result['recall_at_1']:.6f}")
    print(f"Recall@3: {result['recall_at_3']:.6f}")
    print(f"Recall@5: {result['recall_at_5']:.6f}")
    print(f"MRR: {result['mrr']:.6f}")
    if args.output:
        print(f"Details: {project_path(args.output)}")


if __name__ == "__main__":
    main()
