"""Build GOLD documents for our vision artifacts from the SILVER e뮤지엄 corpus.

Matches each artifact in data/metadata.csv (by accession_no <-> SILVER relic_label),
pulls its related records from Silver_relations, and writes one GOLD record per
artifact combining structured fields with a natural-language context blob suitable
for RAG/LLM prompts.

Input:
    data/rag/silver/emuseum/Silver_0928.jsonl
    data/rag/silver/emuseum/Silver_relations_0928.jsonl
    data/metadata.csv

Output:
    data/rag/gold/emuseum/gold_artifacts.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = PROJECT_ROOT / "data" / "rag" / "silver" / "emuseum"
GOLD_DIR = PROJECT_ROOT / "data" / "rag" / "gold" / "emuseum"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"


@dataclass
class MatchResult:
    artifact_id: str
    accession_no: str
    artifact_name: str
    relic: dict | None


def load_metadata_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_silver_records(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return (by_relic_label, by_relic_id) lookup dicts."""
    by_label: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_label[record["relic_label"]] = record
            by_id[record["relic_id"]] = record
    return by_label, by_id


def load_relations(path: Path) -> dict[str, list[dict]]:
    """Return relic_id -> list of relation rows."""
    relations: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            relations.setdefault(record["relic_id"], []).append(record)
    return relations


def match_artifacts(
    metadata_rows: list[dict], silver_by_label: dict[str, dict]
) -> list[MatchResult]:
    results = []
    for row in metadata_rows:
        accession_no = row["accession_no"].strip()
        relic = silver_by_label.get(accession_no)
        results.append(
            MatchResult(
                artifact_id=row["artifact_id"].strip(),
                accession_no=accession_no,
                artifact_name=row["artifact_name"].strip(),
                relic=relic,
            )
        )
    return results


def build_context_text(relic: dict, related_names: list[str]) -> str:
    parts: list[str] = []

    header = relic["name_kr"]
    if relic.get("alt_names"):
        header += f" ({', '.join(relic['alt_names'])})"
    parts.append(header)

    facts: list[str] = []
    if relic.get("nationality") or relic.get("period"):
        facts.append(f"국적/시대: {relic.get('nationality') or '-'} {relic.get('period') or ''}".strip())
    if relic.get("material_l1"):
        material = relic["material_l1"]
        if relic.get("material_l2"):
            material += f" ({relic['material_l2']})"
        facts.append(f"재질: {material}")
    if relic.get("purpose_l1"):
        purpose = " > ".join(
            p for p in [relic.get("purpose_l1"), relic.get("purpose_l2"), relic.get("purpose_l3"), relic.get("purpose_l4")] if p
        )
        facts.append(f"용도: {purpose}")
    if relic.get("designations"):
        designation_line = ", ".join(relic["designations"])
        if relic.get("designation_nos"):
            designation_line += f" ({', '.join(relic['designation_nos'])})"
        facts.append(f"지정: {designation_line}")
    if relic.get("find_place_l1"):
        place = relic["find_place_l1"]
        if relic.get("find_place_l2"):
            place += f" {relic['find_place_l2']}"
        facts.append(f"출토지: {place}")
    if relic.get("size_info"):
        facts.append(f"크기: {relic['size_info']}")
    if facts:
        parts.append(" / ".join(facts))

    if relic.get("description"):
        parts.append(relic["description"])
    else:
        parts.append("(공식 설명문 없음 — 별도 조사 필요)")

    if related_names:
        parts.append("연관 소장품: " + ", ".join(related_names))

    return "\n\n".join(parts)


def build_gold_record(match: MatchResult, relations_by_id: dict[str, list[dict]]) -> dict:
    relic = match.relic
    if relic is None:
        return {
            "artifact_id": match.artifact_id,
            "accession_no": match.accession_no,
            "artifact_name": match.artifact_name,
            "matched": False,
        }

    related = relations_by_id.get(relic["relic_id"], [])
    related_names = [r["related_name"] for r in related if r.get("related_name")]

    return {
        "artifact_id": match.artifact_id,
        "accession_no": match.accession_no,
        "artifact_name": match.artifact_name,
        "matched": True,
        "relic_id": relic["relic_id"],
        "name_kr": relic["name_kr"],
        "alt_names": relic.get("alt_names", []),
        "nationality": relic.get("nationality"),
        "period": relic.get("period"),
        "material_l1": relic.get("material_l1"),
        "material_l2": relic.get("material_l2"),
        "purpose_l1": relic.get("purpose_l1"),
        "purpose_l2": relic.get("purpose_l2"),
        "purpose_l3": relic.get("purpose_l3"),
        "purpose_l4": relic.get("purpose_l4"),
        "designations": relic.get("designations", []),
        "designation_nos": relic.get("designation_nos", []),
        "find_place_l1": relic.get("find_place_l1"),
        "find_place_l2": relic.get("find_place_l2"),
        "size_info": relic.get("size_info"),
        "description": relic.get("description"),
        "related_artifact_names": related_names,
        "image_uri": relic.get("image_uri"),
        "license_code": relic.get("license_code"),
        "context_text": build_context_text(relic, related_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silver-prefix",
        default="0928",
        help="SILVER 파일 날짜 접미사 (기본: 0928, 가장 최신 수집분)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLD_DIR / "gold_artifacts.jsonl",
    )
    args = parser.parse_args()

    silver_path = SILVER_DIR / f"Silver_{args.silver_prefix}.jsonl"
    relations_path = SILVER_DIR / f"Silver_relations_{args.silver_prefix}.jsonl"

    metadata_rows = load_metadata_rows(METADATA_PATH)
    silver_by_label, _silver_by_id = load_silver_records(silver_path)
    relations_by_id = load_relations(relations_path)

    matches = match_artifacts(metadata_rows, silver_by_label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for match in matches:
            record = build_gold_record(match, relations_by_id)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    matched = [m for m in matches if m.relic is not None]
    unmatched = [m for m in matches if m.relic is None]
    no_description = [
        m for m in matched if not (m.relic or {}).get("description")
    ]
    name_mismatch = [
        m
        for m in matched
        if m.relic["name_kr"] != m.artifact_name and m.relic["name_kr"] not in m.relic.get("alt_names", [])
    ]

    print(f"total={len(matches)} matched={len(matched)} unmatched={len(unmatched)}")
    for m in unmatched:
        print(f"  UNMATCHED: {m.artifact_id} ({m.accession_no})")
    for m in no_description:
        print(f"  NO_DESCRIPTION: {m.artifact_id} ({m.accession_no}) name_kr={m.relic['name_kr']!r}")
    for m in name_mismatch:
        print(
            f"  NAME_MISMATCH: {m.artifact_id} metadata.artifact_name={m.artifact_name!r} "
            f"silver.name_kr={m.relic['name_kr']!r}"
        )
    print(f"gold_output={args.output}")


if __name__ == "__main__":
    main()
