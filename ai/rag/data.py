"""GOLD 데이터(chunks, relic_cards, group_cards) 적재와 번호 색인."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from ai.rag.config import GOLD_DIR


Card = dict[str, Any]
Chunk = dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_label(label: str) -> str:
    """'신수 1846' / '신수1846' / '신수  1846' 을 같은 키로."""
    return "".join(label.split())


@dataclass
class GoldData:
    gold_dir: Path = GOLD_DIR
    chunks: list[Chunk] = field(init=False)
    relic_cards: dict[str, Card] = field(init=False)
    group_cards: dict[str, Card] = field(init=False)
    manifest: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.chunks = read_jsonl(self.gold_dir / "chunks.jsonl")
        self.relic_cards = {c["relic_id"]: c for c in read_jsonl(self.gold_dir / "relic_cards.jsonl")}
        self.group_cards = {g["group_id"]: g for g in read_jsonl(self.gold_dir / "group_cards.jsonl")}
        self.manifest = json.loads((self.gold_dir / "_manifest.json").read_text(encoding="utf-8"))

    @cached_property
    def label_index(self) -> dict[str, str]:
        """정규화한 소장품 번호 → relic_id."""
        return {normalize_label(c["relic_label"]): rid for rid, c in self.relic_cards.items()}

    @cached_property
    def chunk_by_id(self) -> dict[str, Chunk]:
        return {c["chunk_id"]: c for c in self.chunks}

    def get_parent(self, parent_type: str, parent_id: str) -> Card:
        return self.relic_cards[parent_id] if parent_type == "relic" else self.group_cards[parent_id]

    def verify_counts(self) -> list[str]:
        """_manifest.json 건수와 적재 건수가 다르면 문제 목록을 반환한다."""
        expected = self.manifest["counts"]
        actual = {
            "chunks": len(self.chunks),
            "relic_cards": len(self.relic_cards),
            "group_and_set_cards": len(self.group_cards),
        }
        wanted = {
            "chunks": expected["chunks"],
            "relic_cards": expected["relic_cards"],
            "group_and_set_cards": expected["group_cards"] + expected["set_cards"],
        }
        return [f"{key}: expected {wanted[key]}, got {actual[key]}" for key in wanted if wanted[key] != actual[key]]
