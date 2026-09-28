"""구조화 Gold 레코드를 검색용 텍스트로 표현한다."""

from langchain_core.documents import Document

FIELD_GROUPS = {
    "basic": [("relic_label", "소장품 번호"), ("name_kr", "유물명"), ("alt_names", "다른 이름"),
              ("author", "작가"), ("period", "시대"), ("nationality", "국적"),
              ("material_l1", "재질 대분류"), ("material_l2", "재질 세부"),
              ("purpose_l1", "용도 대분류"), ("purpose_l2", "용도 2단계"),
              ("purpose_l3", "용도 3단계"), ("purpose_l4", "용도 4단계")],
    "detail": [("size_info", "크기"), ("find_place_l1", "출토지 시도"),
               ("find_place_l2", "출토지 시군구"), ("designations", "지정 종류"),
               ("designation_nos", "지정 번호"), ("description", "설명")],
}


def _line(label: str, value: object) -> str | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, list):
        value = ", ".join(str(x) for x in value if x)
    return f"{label}: {value}" if value else None


def field_sections(row: dict) -> dict[str, str]:
    sections = {}
    for group, fields in FIELD_GROUPS.items():
        lines = [_line(label, row.get(key)) for key, label in fields]
        sections[group] = "\n".join(line for line in lines if line)
    if row.get("is_glass_plate"):
        sections["basic"] += "\n유리건판 사진 기록: 예 (사진의 대상 유물과 구별)"
    relation_lines = [f"관련 유물: {r['related_name']} ({r['related_relic_id']})" for r in row.get("relations", [])]
    sections["relations"] = "\n".join(relation_lines)
    return sections


def build_document(row: dict) -> Document:
    sections = field_sections(row)
    return Document(page_content="\n\n".join(s for s in sections.values() if s),
                    metadata={"relic_id": row["relic_id"], "relic_label": row.get("relic_label", "")})


def build_documents(rows: list[dict]) -> list[Document]:
    return [build_document(row) for row in rows]
