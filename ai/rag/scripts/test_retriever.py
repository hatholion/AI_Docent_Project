"""기존 실험 Chroma에서 Ollama 없이 검색한다."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from langchain_chroma import Chroma
from ai.rag.config import ROOT, load_config
from ai.rag.embeddings import get_embedding
from ai.rag.gold_loader import load_gold, require_relic_label, resolve_relic_id
from ai.rag.retriever import retrieve
from ai.llm.rag_chain import ask_rag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--ask", action="store_true")
    parser.add_argument("--relic-label", help="요청 소장품번호. 생략하면 rag.relic_label 사용")
    args = parser.parse_args()
    config = load_config(args.config)
    selected_label = require_relic_label(args.relic_label if args.relic_label is not None else config.get("rag", {}).get("relic_label"))
    rows = load_gold(ROOT / config["data"]["gold_file"])
    if args.sample is not None:
        if args.sample <= 0:
            parser.error("--sample은 양수여야 합니다")
        rows = rows[:args.sample]
    selected_id = resolve_relic_id(rows, selected_label)
    name = config["experiment"]["name"] + (f"_sample{args.sample}" if args.sample else "")
    path = ROOT / config["vectorstore"]["base_dir"] / name / "chroma"
    if not path.exists():
        parser.error(f"DB가 없습니다: {path}")
    store = Chroma(collection_name=config["vectorstore"]["collection_name"],
                   embedding_function=get_embedding(config["embedding"]), persist_directory=str(path))
    if args.ask:
        print(ask_rag(store, args.query, config["retriever"], config["llm"], relic_id=selected_id))
    else:
        for doc in retrieve(store, args.query, config["retriever"], relic_id=selected_id):
            print(doc.metadata, doc.page_content[:240])


if __name__ == "__main__":
    main()
