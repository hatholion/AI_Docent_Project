"""GOLD 청크 색인 (기획서 §5). 한 번만 실행하면 된다: python -m ai.rag.index

산출물 (data/index/emuseum/, Git 제외):
- chroma/               벡터 색인 (BGE-M3, cosine) + 필터용 metadata
- bm25_tokens.json      키워드 색인 토큰 (chunks.jsonl 순서)
- hanja_readings.tsv    한자 명칭 독음 변환 결과 (검수용)
- index_manifest.json   GOLD 빌드 시각, 임베딩 모델, 건수 검증
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from ai.rag.config import CHROMA_COLLECTION, EMBED_MODEL, INDEX_DIR
from ai.rag.data import Chunk, GoldData
from ai.rag.ollama import embed, model_digest
from ai.rag.text import has_hanja, hanja_reading, tokenize, with_hanja_reading


def search_text(chunk: Chunk) -> str:
    """임베딩·BM25에 넣는 텍스트. GOLD text + 한자 독음(있을 때만)."""
    return with_hanja_reading(chunk["text"])


def chroma_metadata(chunk: Chunk) -> dict[str, Any]:
    """Chroma는 None과 빈 리스트를 저장하지 못하므로 뺀다 (필터에서 '없음'으로 동작)."""
    metadata: dict[str, Any] = {
        "chunk_type": chunk["chunk_type"],
        "parent_type": chunk["parent_type"],
        "parent_id": chunk["parent_id"],
    }
    if chunk.get("relic_label"):
        metadata["relic_label"] = chunk["relic_label"]
    for key, value in chunk["metadata"].items():
        if value is None or value == []:
            continue
        metadata[key] = value
    return metadata


def write_hanja_readings(gold: GoldData, path: Path) -> int:
    rows = [
        (card["relic_label"], card["title"], hanja_reading(card["title"]), ",".join(card["alt_names"]))
        for card in gold.relic_cards.values()
        if has_hanja(card["title"])
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["relic_label", "title", "reading", "alt_names"])
        writer.writerows(rows)
    return len(rows)


def build_index(index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    gold = GoldData()
    problems = gold.verify_counts()
    if problems:
        raise SystemExit(f"GOLD 건수가 _manifest.json과 다릅니다: {problems}")

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)

    texts = [search_text(chunk) for chunk in gold.chunks]
    started = time.perf_counter()
    vectors = embed(texts)
    embed_seconds = time.perf_counter() - started
    print(f"embedded {len(texts)} chunks in {embed_seconds:.1f}s (dim {vectors.shape[1]})")

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    batch = 500
    for start in range(0, len(gold.chunks), batch):
        part = gold.chunks[start : start + batch]
        collection.add(
            ids=[chunk["chunk_id"] for chunk in part],
            embeddings=vectors[start : start + batch].tolist(),
            documents=texts[start : start + batch],
            metadatas=[chroma_metadata(chunk) for chunk in part],
        )

    tokens = [{"chunk_id": chunk["chunk_id"], "tokens": tokenize(text)} for chunk, text in zip(gold.chunks, texts)]
    (index_dir / "bm25_tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    hanja_count = write_hanja_readings(gold, index_dir / "hanja_readings.tsv")

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold_built_at": gold.manifest["built_at"],
        "embedding": {"model": EMBED_MODEL, "digest": model_digest(EMBED_MODEL), "dim": int(vectors.shape[1])},
        "counts": {
            "chunks_indexed": collection.count(),
            "bm25_docs": len(tokens),
            "relic_cards": len(gold.relic_cards),
            "group_and_set_cards": len(gold.group_cards),
            "hanja_titles": hanja_count,
        },
        "embed_seconds": round(embed_seconds, 1),
    }
    if manifest["counts"]["chunks_indexed"] != len(gold.chunks):
        raise SystemExit(f"색인 누락: {manifest['counts']['chunks_indexed']} / {len(gold.chunks)}")
    (index_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    args = parser.parse_args()
    manifest = build_index(args.index_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
