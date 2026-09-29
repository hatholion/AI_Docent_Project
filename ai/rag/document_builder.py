"""Gold 레코드 하나를 검색 문서 하나로 표현한다."""

from __future__ import annotations

from langchain_core.documents import Document

METADATA_LABELS = (
    ("nationality", "국적"),
    ("period", "시대"),
    ("material_l1", "재질 대분류"),
    ("material_l2", "재질 세부"),
    ("purpose_l1", "용도 대분류"),
    ("purpose_l2", "용도 세부"),
    ("find_place_l1", "출토지 광역"),
    ("find_place_l2", "출토지 상세"),
)


def _text_value(value: object) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value if item)
    return str(value) if value else None


def build_document_text(row: dict) -> str:
    """실제 Gold 필드명에 의미 라벨을 붙인다. 식별용 기술 필드는 본문에서 제외한다."""
    metadata = row["metadata"]
    label = _text_value(row.get("relic_label"))
    lines = [
        f"소장품번호: {label}" if label else f"유물 집합 식별자: {row['parent_id']}",
        f"Gold 레코드 유형: {row['chunk_type']}",
    ]
    for key, title in METADATA_LABELS:
        value = _text_value(metadata.get(key))
        if value:
            lines.append(f"{title}: {value}")
    designations = _text_value(metadata.get("designations"))
    if designations:
        lines.append(f"지정 유형: {designations}")
    lines.extend(("", "유물 정보:", row["text"].strip()))
    return "\n".join(lines)


def build_metadata(row: dict) -> dict[str, str | int | float | bool]:
    source = row["metadata"]
    metadata: dict[str, str | int | float | bool] = {
        "chunk_id": row["chunk_id"],
        "chunk_type": row["chunk_type"],
        "parent_type": row["parent_type"],
        "parent_id": row["parent_id"],
        "relic_label": row.get("relic_label") or "",
        "collection_name": source.get("collection_name") or "",
        "record_type": source.get("record_type") or "",
    }
    for key in ("nationality", "period", "material_l1", "material_l2",
                "purpose_l1", "purpose_l2", "find_place_l1", "find_place_l2"):
        value = source.get(key)
        if isinstance(value, (str, int, float, bool)) and value != "":
            metadata[key] = value
    designations = source.get("designations")
    if isinstance(designations, list) and designations:
        metadata["designations"] = ", ".join(str(item) for item in designations if item)
    return metadata


def build_document(row: dict) -> Document:
    return Document(
        id=row["chunk_id"],
        page_content=build_document_text(row),
        metadata=build_metadata(row),
    )


def build_documents(rows: list[dict] | tuple[dict, ...]) -> list[Document]:
    return [build_document(row) for row in rows]
