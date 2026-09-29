"""완성된 Gold JSONL을 변경 없이 읽고 직접 조회한다."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

REQUIRED_FIELDS = {
    "chunk_id", "chunk_type", "parent_type", "parent_id",
    "relic_label", "text", "char_len", "metadata",
}
REQUIRED_METADATA_FIELDS = {
    "record_type", "collection_name", "nationality", "period",
    "material_l1", "material_l2", "purpose_l1", "purpose_l2",
    "find_place_l1", "find_place_l2", "designations",
    "has_description", "has_image",
}


class RelicNotFoundError(LookupError):
    pass


def require_relic_label(relic_label: str | None) -> str:
    if not isinstance(relic_label, str) or not relic_label.strip():
        raise ValueError("relic_label은 비어 있지 않은 문자열이어야 합니다")
    return relic_label.strip()


def _validate_row(row: object, number: int, seen: set[str]) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"Gold {number}행: JSON 객체가 필요합니다")
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"Gold {number}행: 필드 누락: {', '.join(missing)}")
    chunk_id = row["chunk_id"]
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(f"Gold {number}행: chunk_id가 비어 있습니다")
    if chunk_id in seen:
        raise ValueError(f"Gold {number}행: chunk_id 중복: {chunk_id}")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError(f"Gold {number}행: text가 비어 있습니다")
    for key in ("chunk_type", "parent_type", "parent_id"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"Gold {number}행: {key}가 비어 있습니다")
    label = row["relic_label"]
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError(f"Gold {number}행: relic_label은 문자열 또는 null이어야 합니다")
    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Gold {number}행: metadata 객체가 필요합니다")
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing_metadata:
        raise ValueError(f"Gold {number}행: metadata 필드 누락: {', '.join(missing_metadata)}")
    seen.add(chunk_id)
    return row


def load_gold(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Gold dataset을 찾을 수 없습니다: {source}")
    rows: list[dict] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Gold {number}행: 올바른 JSON이 아닙니다") from exc
            rows.append(_validate_row(row, number, seen))
    if not rows:
        raise ValueError("Gold dataset이 비어 있습니다")
    return rows


@lru_cache(maxsize=4)
def _cached_rows(path: Path) -> tuple[dict, ...]:
    return tuple(load_gold(path))


def cached_gold(path: str | Path) -> tuple[dict, ...]:
    return _cached_rows(Path(path).resolve())


@lru_cache(maxsize=4)
def _label_index(path: Path) -> dict[str, tuple[dict, ...]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _cached_rows(path):
        label = row.get("relic_label")
        if isinstance(label, str) and label.strip():
            grouped[label.strip()].append(row)
    return {label: tuple(records) for label, records in grouped.items()}


def get_relic_records_by_label(path: str | Path, relic_label: str | None) -> list[dict]:
    """같은 유물의 profile/description Gold 레코드를 모두 반환한다."""
    label = require_relic_label(relic_label)
    records = _label_index(Path(path).resolve()).get(label)
    if not records:
        raise RelicNotFoundError(f"Gold에 relic_label이 없습니다: {label}")
    return list(records)


def gold_summary(rows: list[dict] | tuple[dict, ...]) -> dict:
    labels = [row.get("relic_label") for row in rows]
    return {
        "records": len(rows),
        "unique_chunk_ids": len({row["chunk_id"] for row in rows}),
        "unique_parent_ids": len({row["parent_id"] for row in rows}),
        "unique_relic_labels": len({label for label in labels if label}),
        "missing_relic_labels": sum(not label for label in labels),
        "empty_documents": sum(not row["text"].strip() for row in rows),
    }


def clear_gold_cache() -> None:
    _cached_rows.cache_clear()
    _label_index.cache_clear()
