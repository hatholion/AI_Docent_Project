"""Pre-generate visitor-type-adapted descriptions for every artifact in GOLD.

Reads data/rag/gold/emuseum/gold_artifacts.jsonl (7 artifacts) and, for each
artifact, generates child/general/expert descriptions grounded in its
context_text via a local Ollama model. Results are cached to a JSONL file so
the backend can serve them without calling the LLM at request time.

Output: data/rag/gold/emuseum/descriptions_by_type.jsonl
    {artifact_id, visitor_type, text, model, generated_at}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.llm.ollama_client import generate
from ai.rag.prompts import VISITOR_INSTRUCTIONS, build_prompt


GOLD_PATH = PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum" / "gold_artifacts.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum" / "descriptions_by_type.jsonl"


def load_gold_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="exaone3.5:7.8b")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = load_gold_records(GOLD_PATH)
    if not records:
        raise SystemExit(f"No GOLD records found in {GOLD_PATH}")

    results = []
    for record in records:
        artifact_id = record["artifact_id"]
        context_text = record["context_text"]
        for visitor_type in VISITOR_INSTRUCTIONS:
            prompt = build_prompt(context_text, visitor_type)
            start = time.time()
            response = generate(prompt, args.model)
            elapsed = time.time() - start
            text = response.get("response", "").strip()
            results.append(
                {
                    "artifact_id": artifact_id,
                    "visitor_type": visitor_type,
                    "text": text,
                    "model": args.model,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"{artifact_id}/{visitor_type}: elapsed={elapsed:.2f}s chars={len(text)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"total={len(results)} output={args.output}")


if __name__ == "__main__":
    main()
