"""Smoke-test visitor-type prompts against a local Ollama model, grounded in GOLD data.

Prints results as UTF-8 JSON to --output so Windows console codepage issues
don't mangle the Korean text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.llm.ollama_client import generate
from ai.rag.prompts import VISITOR_INSTRUCTIONS, build_prompt

GOLD_PATH = PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum" / "gold_artifacts.jsonl"


def load_gold_record(artifact_id: str) -> dict:
    with GOLD_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["artifact_id"] == artifact_id:
                return record
    raise KeyError(f"artifact_id not found in GOLD: {artifact_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--model", default="exaone3.5:7.8b")
    parser.add_argument("--output", type=Path, default=Path("docent_prompt_result.json"))
    args = parser.parse_args()

    record = load_gold_record(args.artifact_id)
    context_text = record["context_text"]

    results = {"artifact_id": args.artifact_id, "model": args.model, "by_visitor_type": {}}
    for visitor_type in VISITOR_INSTRUCTIONS:
        prompt = build_prompt(context_text, visitor_type)
        start = time.time()
        response = generate(prompt, args.model)
        elapsed = time.time() - start
        results["by_visitor_type"][visitor_type] = {
            "prompt": prompt,
            "response": response.get("response"),
            "elapsed_sec": round(elapsed, 2),
            "eval_count": response.get("eval_count"),
        }
        print(f"{visitor_type}: elapsed={elapsed:.2f}s eval_count={response.get('eval_count')}")

    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
