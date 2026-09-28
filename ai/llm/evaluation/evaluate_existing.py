"""기존 Chroma DB를 재인덱싱하지 않고 청크 정답셋을 평가한다."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from langchain_chroma import Chroma
from ai.rag.config import ROOT, load_config
from ai.rag.embeddings import get_embedding
from ai.rag.gold_loader import load_gold
from ai.llm.evaluation.retrieval_eval import evaluate, load_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", help="실제 라벨 JSON. 생략하면 evaluation.dataset 사용")
    args = parser.parse_args()
    config = load_config(args.config)
    name = config["experiment"]["name"]
    db_path = ROOT / config["vectorstore"]["base_dir"] / name / "chroma"
    if not db_path.exists():
        parser.error(f"DB가 없습니다: {db_path}")
    dataset_path = ROOT / (args.dataset or config["evaluation"]["dataset"])
    if not dataset_path.exists():
        parser.error(f"실제 정답셋이 없습니다: {dataset_path}")
    store = Chroma(collection_name=config["vectorstore"]["collection_name"],
                   embedding_function=get_embedding(config["embedding"]),
                   persist_directory=str(db_path))
    result = evaluate(store, load_evaluation(dataset_path), config["retriever"],
                      gold_rows=load_gold(ROOT / config["data"]["gold_file"]))
    path = ROOT / "ai/llm/evaluation/results" / f"{name}_retrieval_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
