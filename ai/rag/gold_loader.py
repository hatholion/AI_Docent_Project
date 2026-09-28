"""Gold JSONL만 읽는 RAG 데이터 입력점."""

import json
from pathlib import Path


def require_relic_label(relic_label: str | None) -> str:
    """소장품번호는 앞뒤 공백만 제거하며 내부 공백은 유지한다."""
    if not isinstance(relic_label, str) or not relic_label.strip():
        raise ValueError("relic_label은 필수입니다. --relic-label 또는 rag.relic_label에 지정하세요")
    return relic_label.strip()


def resolve_relic_id(rows: list[dict], relic_label: str | None) -> str:
    """Gold 범위에서 소장품번호가 유일하게 가리키는 내부 ID를 찾는다."""
    label = require_relic_label(relic_label)
    matches = [row for row in rows
               if isinstance(row.get("relic_label"), str) and row["relic_label"].strip() == label]
    if not matches:
        raise ValueError(f"Gold 범위에 relic_label이 없습니다: {label}")
    if len(matches) != 1:
        raise ValueError(f"Gold에서 relic_label이 중복됩니다: {label}")
    relic_id = matches[0].get("relic_id")
    if not isinstance(relic_id, str) or not relic_id.strip():
        raise ValueError(f"Gold의 relic_id가 올바르지 않습니다: {label}")
    return relic_id


def load_gold(path: str | Path) -> list[dict]:
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            relic_id = row.get("relic_id")
            if not relic_id or relic_id in seen:
                raise ValueError(f"Gold {number}행: relic_id 누락 또는 중복")
            seen.add(relic_id)
            rows.append(row)
    return rows
