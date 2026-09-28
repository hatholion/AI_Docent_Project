"""Silver 세 파일을 유물 단위 Gold JSONL로 결합한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/silver"
LOG = logging.getLogger(__name__)

# 동일 명칭의 서로 다른 유물 ID는 모두 포함한다. 띄어쓰기 차이는 같은 이름으로 본다.
TARGET_NAMES = (
    "금동 반가사유상", "백자 달항아리", "요령식 동검", "빗살무늬 토기",
    "방패형 동기", "농경문 청동기", "청동 촛대",
)

TARGET_IDS = (
    "본관 2789", "덕수 798", "접수 702", "신수 1794",
    "신수 1846", "신수 3094", "신수 22891",
)


def normalized_name(name: str) -> str:
    #"""유물명의 공백 차이만 정규화한다."""
    #return re.sub(r"\s+", "", name)
    """양 사이드 공백만 제거"""
    return name.strip()


def select_artifacts(rows: list[dict]) -> list[dict]:
    '''
    """지정한 일곱 명칭에 해당하는 모든 유물 ID를 선택한다."""
    targets = {normalized_name(name) for name in TARGET_NAMES}
    selected = [row for row in rows if normalized_name(row.get("name_kr") or "") in targets]
    '''
    """지정한 일곱 소장품번호에 해당하는 모든 유물 ID를 선택한다."""
    targets = {normalized_name(id) for id in TARGET_IDS}
    selected = [row for row in rows if normalized_name(row.get("relic_label") or "") in targets]  
    found = {normalized_name(row["relic_label"]) for row in selected}
    if missing := targets - found:
        raise ValueError(f"Silver에서 대상 유물명을 찾지 못했습니다: {sorted(missing)}")
    return selected


def load_jsonl(path: Path) -> list[dict]:
    """줄 번호를 포함해 잘못된 JSON을 보고한다."""
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: object가 아닙니다")
                rows.append(row)
    return rows


def variable_names(path: Path) -> list[str]:
    """var_list.md 표의 첫 열에서 변수명을 순서대로 읽는다."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*`([a-z][a-z0-9_]*)`\s*\|", line)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    if not names:
        raise ValueError(f"변수 표를 찾지 못했습니다: {path}")
    return names


def build_gold(artifact_rows: list[dict], image_rows: list[dict],
               relation_rows: list[dict], fields: list[str]) -> list[dict]:
    """relic_id를 기준으로 1:N 자식 행을 순서대로 묶는다."""
    if "relic_id" not in fields:
        raise ValueError("var_list.md에 relic_id가 필요합니다")
    if not artifact_rows:
        raise ValueError("본체 Silver에 유물 행이 없습니다")
    ids = [row.get("relic_id") for row in artifact_rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("본체 Silver의 relic_id가 없거나 중복되었습니다")
    missing = set(fields) - set.intersection(*(set(row) for row in artifact_rows))
    if missing:
        raise ValueError(f"본체 Silver에 없는 필드: {sorted(missing)}")
    known = set(ids)
    groups = []
    for rows, label in ((image_rows, "image"), (relation_rows, "relation")):
        grouped = defaultdict(list)
        for row in rows:
            relic_id = row.get("relic_id")
            if not relic_id or relic_id not in known:
                raise ValueError(f"{label}의 연결할 수 없는 relic_id: {relic_id}")
            grouped[relic_id].append(row)
        groups.append(grouped)
    images, relations = groups
    output = []
    for row in artifact_rows:
        relic_id = row["relic_id"]
        item = {field: row[field] for field in fields}
        item["images"] = sorted(images[relic_id], key=lambda x: (x["image_seq"], x["img_order"], x["image_id"]))
        item["relations"] = sorted(relations[relic_id], key=lambda x: (x["relation_order"], x["related_relic_id"]))
        output.append(item)
    return output


def summarize(rows: list[dict], fields: list[str]) -> None:
    ids = [row.get("relic_id") for row in rows]
    counts = Counter(ids)
    print(f"Gold records: {len(rows)}")
    print(f"Unique artifacts: {len(set(x for x in ids if x))}")
    print(f"Duplicate artifact IDs: {sum(n - 1 for n in counts.values() if n > 1)}")
    print(f"Missing artifact IDs: {sum(not x for x in ids)}")
    for field in [*fields, "images", "relations"]:
        nonempty = sum(row[field] is not None and row[field] != "" and row[field] != [] for row in rows)
        print(f"{field}: {nonempty}/{len(rows)} non-empty")
    print(f"Image rows aggregated: {sum(len(r['images']) for r in rows)}")
    print(f"Relation rows aggregated: {sum(len(r['relations']) for r in rows)}")
    for row in rows[:3]:
        print("Sample:", json.dumps({k: v for k, v in row.items() if k not in ("images", "relations", "description")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/gold/Gold_0928.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and not args.overwrite:
        parser.error(f"이미 존재합니다: {output} (--overwrite 필요)")
    fields = variable_names(DATA / "var_list.md")
    artifact_rows = select_artifacts(load_jsonl(DATA / "Silver_0928.jsonl"))
    selected_ids = {row["relic_id"] for row in artifact_rows}
    image_rows = [row for row in load_jsonl(DATA / "Silver_images_0928.jsonl")
                  if row.get("relic_id") in selected_ids]
    relation_rows = [row for row in load_jsonl(DATA / "Silver_relations_0928.jsonl")
                     if row.get("relic_id") in selected_ids]
    #for name in TARGET_NAMES:
    for id in TARGET_IDS:
        count = sum(normalized_name(row["relic_label"]) == normalized_name(id) for row in artifact_rows)
        LOG.info("대상 유물 %s: %d건", id, count)
    rows = build_gold(artifact_rows, image_rows, relation_rows, fields)
    summarize(rows, fields)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    LOG.info("Gold 저장: %s", output)


if __name__ == "__main__":
    main()
