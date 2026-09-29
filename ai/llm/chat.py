"""최초 유물 설명과 단일 후속 질문을 실행하는 로컬 CLI."""

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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ai/llm/configs/base.yaml")
    parser.add_argument("--relic-label")
    parser.add_argument("--session-id")
    parser.add_argument("--question", help="지정하면 최초 설명 뒤 후속 질문까지 실행")
    parser.add_argument("--visitor-type", choices=("child", "general", "expert"))
    args = parser.parse_args()
    service = make_service(load_config(args.config))
    initial = service.describe_relic(
        relic_label=args.relic_label,
        session_id=args.session_id,
        visitor_type=args.visitor_type,
    )
    output = {"initial": initial}
    if args.question:
        output["follow_up"] = service.answer_follow_up(
            args.question,
            session_id=initial["session_id"],
            visitor_type=args.visitor_type,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
