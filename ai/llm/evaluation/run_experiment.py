"""고정 Gold로 Chroma를 구축하고 정답셋이 있으면 검색 평가한다."""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ai.rag.config import ROOT, load_config
from ai.rag.gold_loader import load_gold, require_relic_label, resolve_relic_id
from ai.rag.document_builder import build_documents
from ai.rag.chunkers import chunk_documents
from ai.rag.embeddings import get_embedding
from ai.rag.vector_store import create_chroma
from ai.rag.retriever import retrieve
from ai.llm.evaluation.retrieval_eval import load_evaluation, evaluate


def run(config_file: str, sample: int | None, rebuild: bool,
        relic_label: str | None = None) -> dict:
    config = load_config(config_file)
    selected_label = require_relic_label(relic_label if relic_label is not None else config.get("rag", {}).get("relic_label"))
    rows = load_gold(ROOT / config["data"]["gold_file"])
    if sample is not None:
        if sample <= 0:
            raise ValueError("--sample은 양수여야 합니다")
        rows = rows[:sample]
    selected_id = resolve_relic_id(rows, selected_label)
    documents = build_documents(rows)
    chunks = chunk_documents(documents, config["chunking"], rows)
    embedding = get_embedding(config["embedding"])
    name = config["experiment"]["name"] + (f"_sample{sample}" if sample else "")
    vector_config = dict(config["vectorstore"])
    vector_config["base_dir"] = str(ROOT / vector_config["base_dir"])
    store = create_chroma(chunks, embedding, vector_config, name, rebuild)
    manifest_file = ROOT / "ai/llm/evaluation/results" / f"{name}_chunks.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps({"chunk_id": chunk.metadata["chunk_id"],
                                     "relic_id": chunk.metadata["relic_id"],
                                     "chunk_index": chunk.metadata["chunk_index"],
                                     "chunk_type": chunk.metadata["chunk_type"],
                                     "page_content": chunk.page_content}, ensure_ascii=False) + "\n")
    result = {"experiment": name, "gold_records": len(rows), "chunks": len(chunks),
              "relic_id": selected_id,
              "relic_label": selected_label,
              "gold_file": config["data"]["gold_file"],
              "chunk_manifest": str(manifest_file.relative_to(ROOT))}
    dataset = ROOT / config["evaluation"]["dataset"]
    if sample is not None:
        logging.info("샘플 DB에서는 전체 정답셋 평가를 건너뜁니다")
    elif dataset.exists():
        result["evaluation"] = evaluate(store, load_evaluation(dataset), config["retriever"], gold_rows=rows)
    else:
        logging.warning("정답 평가셋이 없어 지표를 건너뜁니다: %s", dataset)
    query = config.get("rag", {}).get("smoke_query")
    if query:
        result["smoke_query"] = query
        smoke_docs = retrieve(store, query, config["retriever"], relic_id=selected_id)
        result["smoke_ids"] = [d.metadata["relic_id"] for d in smoke_docs]
        result["smoke_chunk_ids"] = [d.metadata["chunk_id"] for d in smoke_docs]
    result_file = ROOT / "ai/llm/evaluation/results" / f"{name}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("결과 저장: %s", result_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--relic-label", help="로컬 검색 소장품번호. 생략하면 rag.relic_label 사용")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(args.config, args.sample, args.rebuild, args.relic_label), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
