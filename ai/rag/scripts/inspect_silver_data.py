"""Silver 스키마와 연결 cardinality를 간단히 확인한다."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ai.rag.scripts.build_gold_dataset import DATA, load_jsonl, variable_names


def main() -> None:
    fields = variable_names(DATA / "var_list.md")
    for name in ("Silver_0923.jsonl", "Silver_images_0923.jsonl", "Silver_relations_0923.jsonl"):
        rows = load_jsonl(DATA / name)
        ids = Counter(row.get("relic_id") for row in rows)
        print(f"{name}: rows={len(rows)}, artifacts={len(ids)}, max_rows_per_artifact={max(ids.values())}")
        print("fields:", sorted(rows[0]) if rows else [])
        print("var_list fields:", [field for field in fields if rows and field in rows[0]])


if __name__ == "__main__":
    main()
