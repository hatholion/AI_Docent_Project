"""Read-only access to GOLD documents and pre-generated descriptions.

Both files are produced by ai/rag/build_gold.py and ai/rag/generate_descriptions.py
and live under data/rag/ (Git-ignored, distributed via the team's shared drive).
Loaded lazily and cached so a missing file doesn't break the whole backend at
import time - only the docent endpoints that need it will fail.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ai.vision.config import PROJECT_ROOT


GOLD_PATH = PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum" / "gold_artifacts.jsonl"
DESCRIPTIONS_PATH = (
    PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum" / "descriptions_by_type.jsonl"
)


class GoldDataMissing(LookupError):
    """GOLD/description 파일 또는 그 안의 항목을 찾을 수 없을 때."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise GoldDataMissing(
            f"{path} 없음. ai/rag/build_gold.py, ai/rag/generate_descriptions.py로 "
            "생성하거나 팀 공유 드라이브에서 data/rag/를 받아오세요."
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=1)
def _gold_by_artifact_id() -> dict[str, dict]:
    return {record["artifact_id"]: record for record in _read_jsonl(GOLD_PATH)}


@lru_cache(maxsize=1)
def _descriptions_by_key() -> dict[tuple[str, str], dict]:
    return {
        (record["artifact_id"], record["visitor_type"]): record
        for record in _read_jsonl(DESCRIPTIONS_PATH)
    }


def get_context_text(artifact_id: str) -> str:
    record = _gold_by_artifact_id().get(artifact_id)
    if record is None:
        raise GoldDataMissing(f"GOLD 문서에 artifact_id '{artifact_id}' 없음")
    return record["context_text"]


def get_cached_description(artifact_id: str, visitor_type: str) -> str:
    record = _descriptions_by_key().get((artifact_id, visitor_type))
    if record is None:
        raise GoldDataMissing(
            f"사전 생성된 설명 없음: artifact_id '{artifact_id}', visitor_type '{visitor_type}'"
        )
    return record["text"]
