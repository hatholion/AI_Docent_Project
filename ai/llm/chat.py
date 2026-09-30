"""최초 유물 설명과 단일 후속 질문을 실행하는 로컬 CLI.

기본 출력은 사용자에게 보여줄 답변 문자열만 사용한다. 세션 ID, 검색 결과 등
디버깅용 구조가 필요할 때는 ``--json``을 지정한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.rag.chat import DocentChatService
from ai.rag.config import load_config, project_path
from ai.rag.embeddings import get_embedding
from ai.rag.vector_store import open_chroma


def make_service(config: dict) -> DocentChatService:
    vector = dict(config["vector_db"])
    vector["persist_directory"] = str(project_path(vector["persist_directory"]))
    vector["embedding_model"] = config["embedding"]["model"]
    store = open_chroma(vector, get_embedding(config["embedding"]))
    runtime = dict(config)
    runtime["data"] = {**config["data"], "gold_path": str(project_path(config["data"]["gold_path"]))}
    return DocentChatService(config=runtime, store=store)


def run_interactive(service: DocentChatService, initial: dict, visitor_type: str | None) -> None:
    """같은 세션을 유지하며 사용자의 후속 질문을 반복해서 처리한다."""
    session_id = initial["session_id"]
    print(initial["answer"], flush=True)
    while True:
        try:
            question = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit", "종료"}:
            return
        try:
            result = service.answer_follow_up(
                question,
                session_id=session_id,
                visitor_type=visitor_type,
            )
        except Exception as exc:
            print(f"오류: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"\n{result['answer']}", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ai/llm/configs/base.yaml")
    parser.add_argument("--relic-label")
    parser.add_argument("--session-id")
    parser.add_argument("--question", help="지정하면 최초 설명 뒤 후속 질문까지 실행")
    parser.add_argument("--visitor-type", choices=("child", "general", "expert"))
    parser.add_argument("--json", action="store_true", help="답변 대신 전체 결과를 JSON으로 출력")
    parser.add_argument("--interactive", action="store_true", help="직접 질문을 입력하는 멀티턴 대화 모드")
    args = parser.parse_args()
    if args.interactive and args.question:
        parser.error("--interactive와 --question은 함께 사용할 수 없습니다")
    if args.interactive and args.json:
        parser.error("--interactive와 --json은 함께 사용할 수 없습니다")
    service = make_service(load_config(args.config))
    initial = service.describe_relic(
        relic_label=args.relic_label,
        session_id=args.session_id,
        visitor_type=args.visitor_type,
    )
    if args.interactive:
        run_interactive(service, initial, args.visitor_type)
        return
    output = {"initial": initial}
    if args.question:
        output["follow_up"] = service.answer_follow_up(
            args.question,
            session_id=initial["session_id"],
            visitor_type=args.visitor_type,
        )
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif args.question:
        print(output["follow_up"]["answer"])
    else:
        print(output["initial"]["answer"])


if __name__ == "__main__":
    main()
