"""질문 한 개를 검색·답변까지 돌려보는 CLI.

python -m ai.rag.ask "신수 1846은 무엇인가요?"
python -m ai.rag.ask "조선 백자 중 보물" --no-llm      # 검색 결과만
"""

from __future__ import annotations

import argparse
import time

from ai.rag.generate import generate_answer
from ai.rag.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--visitor-type", choices=("child", "general", "expert"), default="general")
    parser.add_argument("--no-llm", action="store_true", help="답변 생성 없이 검색 결과만 출력")
    args = parser.parse_args()

    retriever = Retriever()
    started = time.perf_counter()
    result = retriever.retrieve(args.question)
    print(f"route={result.route} filters={result.filters} found={result.found} "
          f"best_vec={result.best_vector_similarity:.3f} best_bm25={result.best_bm25_score:.2f} "
          f"({time.perf_counter() - started:.2f}s)")
    for rank, parent in enumerate(result.parents, 1):
        print(f"  {rank}. [{parent.via}] {parent.label or parent.parent_id} {parent.card['title']} ({parent.score:.4f})")
    if args.no_llm:
        return

    answer = generate_answer(result, visitor_type=args.visitor_type)
    print(f"\n{answer.answer}\n")
    for source in answer.sources:
        print(f"출처: {source}")
    if answer.removed_labels:
        print(f"(자료에 없는 번호 제거: {answer.removed_labels})")
    print(f"(LLM {answer.llm_seconds}s)")


if __name__ == "__main__":
    main()
