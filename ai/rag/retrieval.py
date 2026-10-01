"""검색 (기획서 §6): 질문 라우팅 → 하이브리드 검색 → parent 변환 → 연관 확장."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from ai.rag.config import CHROMA_COLLECTION, INDEX_DIR, RetrievalSettings
from ai.rag.data import Card, GoldData, normalize_label
from ai.rag.ollama import embed
from ai.rag.text import tokenize, with_hanja_reading


# GOLD README §4.2의 소장품 번호 패턴. 한 글자 구분(구·안·증)은 번호 색인에 실제 있는 값만 인정한다.
LABEL_RE = re.compile(
    r"(본관|개성|남산|접수|덕수|수정|신안|신도|동원|정내|신수|M번|K번|근대|구|증|홍산|건판|고적|안|별관|건희)\s?(\d+)(-\d+)?"
)

# 조건 검색 사전 (1차: 사전 매칭). 일반 명사와 겹치는 값은 '시대'가 붙을 때만 인정한다.
PERIODS = [
    "통일신라", "초기철기", "원삼국", "대한제국", "일제강점", "려말선초", "신석기", "구석기",
    "고구려", "조선", "고려", "가야", "신라", "백제", "낙랑", "삼국",
]
PERIODS_NEED_SUFFIX = ["청동기"]
MATERIALS_L2 = ["백자", "청자", "분청", "금동", "흑유", "천목", "화강암", "대리석", "동합금"]
MATERIALS_L1 = ["도자기", "금속", "토제", "나무", "유리/보석"]
DESIGNATIONS = ["국가민속문화유산", "등록문화유산", "시도지정유산", "국보", "보물"]
PHOTO_WORDS = ["유리건판", "건판", "사진"]
STATS_RE = re.compile(r"몇\s?(점|개|건|종|가지)|총\s?몇|전체\s?(수|개수|규모)|통계|분포|비율|얼마나\s?많")
# "비암사 비상은 몇 점으로 이루어져 있어?" 같은 세트 구성 질문은 답할 수 있으므로,
# 소장품 전체 범위를 가리키는 말이 함께 있을 때만 통계 질문으로 본다
STATS_SCOPE_RE = re.compile(r"박물관|소장품|소장|전체|전부|총|모두|통계|분포|비율")


@dataclass
class RetrievedParent:
    parent_type: str  # relic / group / set
    parent_id: str
    card: Card
    score: float
    via: str  # label / search / related / set

    @property
    def label(self) -> str | None:
        return self.card.get("relic_label")

    @property
    def citation(self) -> str:
        if self.parent_type == "relic":
            return self.card["citation"]
        # 묶음·세트 카드는 context_text 마지막 줄이 출처
        last = self.card["context_text"].strip().splitlines()[-1]
        return last.removeprefix("출처:").strip()


@dataclass
class RetrievalResult:
    question: str
    route: str  # label / search
    filters: dict[str, Any]
    parents: list[RetrievedParent]
    found: bool
    stats_question: bool
    best_vector_similarity: float = 0.0
    best_bm25_score: float = 0.0
    child_hits: list[tuple[str, float]] = field(default_factory=list)
    # True면 LLM 없이 찾은 카드 목록으로 답한다 (채팅의 탐색 질문. 모델이 자료 밖 유물을 지어내는 것을 막는다)
    list_answer: bool = False

    def to_contexts(self) -> list[dict[str, str]]:
        """팀 공용 RAG 결과 형식 (Git_규칙 14): [{text, source}]."""
        return [{"text": p.card["context_text"], "source": p.citation} for p in self.parents]


def is_stats_question(question: str) -> bool:
    return bool(STATS_RE.search(question)) and bool(STATS_SCOPE_RE.search(question))


def extract_filters(question: str) -> dict[str, Any]:
    """질문에서 시대·재질·지정 조건을 뽑는다. 기본은 record_type = artifact."""
    filters: dict[str, Any] = {}
    remaining = question
    for period in PERIODS:  # 긴 값부터 (통일신라가 신라보다 먼저)
        if period in remaining:
            filters["period"] = period
            remaining = remaining.replace(period, " ")
            break
    else:
        for period in PERIODS_NEED_SUFFIX:
            if re.search(rf"{period}\s?시대", remaining):
                filters["period"] = period
                break
    for material in MATERIALS_L2:
        if material in question:
            filters["material_l2"] = material
            break
    else:
        for material in MATERIALS_L1:
            if material in question:
                filters["material_l1"] = material
                break
    for designation in DESIGNATIONS:
        if designation in question:
            filters["designations"] = designation
            break
    if not any(word in question for word in PHOTO_WORDS):
        filters["record_type"] = "artifact"
    return filters


def chroma_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    clauses = [
        {key: {"$contains": value}} if key == "designations" else {key: value}
        for key, value in filters.items()
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if key == "designations":
            if value not in (metadata.get("designations") or []):
                return False
        elif metadata.get(key) != value:
            return False
    return True


def expand_parents(gold: GoldData, parents: list[RetrievedParent], related_max: int) -> list[RetrievedParent]:
    """세트 카드와 1순위 카드의 연관 소장품을 더한다. 색인 없이 카드만으로 동작한다 (ID 조회에서도 사용)."""
    seen = {p.parent_id for p in parents}
    extra: list[RetrievedParent] = []
    # 세트 구성품이 걸리면 세트 카드를 함께 넣는다
    for parent in parents:
        set_id = parent.card.get("set_id")
        if parent.parent_type == "relic" and set_id and set_id not in seen:
            extra.append(RetrievedParent("set", set_id, gold.group_cards[set_id], parent.score, "set"))
            seen.add(set_id)
    # 1순위 카드의 연관 소장품 중 수집된 것 최대 N장 (연관의 연관은 가져오지 않음)
    if parents and parents[0].parent_type == "relic":
        added = 0
        for related in parents[0].card.get("related", []):
            if added >= related_max:
                break
            relic_id = related["relic_id"]
            if related.get("collected") and relic_id not in seen and relic_id in gold.relic_cards:
                extra.append(RetrievedParent("relic", relic_id, gold.relic_cards[relic_id], 0.0, "related"))
                seen.add(relic_id)
                added += 1
    return parents + extra


class Retriever:
    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        *,
        gold: GoldData | None = None,
        index_dir: Path = INDEX_DIR,
    ) -> None:
        self.settings = settings or RetrievalSettings()
        self.gold = gold or GoldData()
        self.index_dir = index_dir
        if not (index_dir / "index_manifest.json").is_file():
            raise FileNotFoundError(f"색인이 없습니다. 먼저 `python -m ai.rag.index`를 실행하세요: {index_dir}")
        self.index_manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))

    @cached_property
    def collection(self):
        return chromadb.PersistentClient(path=str(self.index_dir / "chroma")).get_collection(CHROMA_COLLECTION)

    @cached_property
    def _bm25(self) -> tuple[BM25Okapi, list[str]]:
        entries = json.loads((self.index_dir / "bm25_tokens.json").read_text(encoding="utf-8"))
        return BM25Okapi([entry["tokens"] or ["_"] for entry in entries]), [entry["chunk_id"] for entry in entries]

    # --- 1. 라우팅 -------------------------------------------------------
    def find_labels(self, question: str) -> tuple[list[str], list[str]]:
        """(번호 색인에 있는 relic_id, 형식은 번호인데 색인에 없는 번호)."""
        relic_ids: list[str] = []
        unknown: list[str] = []
        for match in LABEL_RE.finditer(question):
            relic_id = self.gold.label_index.get(normalize_label(match.group(0)))
            if relic_id and relic_id not in relic_ids:
                relic_ids.append(relic_id)
            elif not relic_id and len(match.group(1)) >= 2:
                # 구·안·증 같은 한 글자 구분은 일반 문장과 겹치므로 '없는 번호'로 보지 않는다
                unknown.append(match.group(0))
        return relic_ids, unknown

    # --- 2. 하이브리드 검색 ----------------------------------------------
    def vector_search(self, question: str, filters: dict[str, Any], k: int) -> list[tuple[str, float]]:
        query = embed([with_hanja_reading(question)])[0]
        result = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
            where=chroma_where(filters),
            include=["distances"],
        )
        return [(chunk_id, 1.0 - distance) for chunk_id, distance in zip(result["ids"][0], result["distances"][0])]

    def bm25_search(self, question: str, filters: dict[str, Any], k: int) -> list[tuple[str, float]]:
        bm25, chunk_ids = self._bm25
        tokens = tokenize(with_hanja_reading(question))
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        order = np.argsort(-scores)
        hits: list[tuple[str, float]] = []
        for index in order:
            if scores[index] <= 0 or len(hits) >= k:
                break
            chunk = self.gold.chunk_by_id[chunk_ids[index]]
            if matches(chunk["metadata"], filters):
                hits.append((chunk_ids[index], float(scores[index])))
        return hits

    def hybrid_search(self, question: str, filters: dict[str, Any]) -> tuple[dict[str, float], float, float]:
        k = self.settings.k_child
        vector_hits = self.vector_search(question, filters, k)
        bm25_hits = self.bm25_search(question, filters, k)
        fused: dict[str, float] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, (chunk_id, _) in enumerate(hits):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.settings.rrf_k + rank + 1)
        best_vector = vector_hits[0][1] if vector_hits else 0.0
        best_bm25 = bm25_hits[0][1] if bm25_hits else 0.0
        return fused, best_vector, best_bm25

    # --- 3. parent 변환 / 4. 연관 확장 -----------------------------------
    def to_parents(self, fused: dict[str, float], exclude_ids: set[str] | None = None) -> list[RetrievedParent]:
        best: dict[tuple[str, str], float] = {}
        for chunk_id, score in fused.items():
            chunk = self.gold.chunk_by_id[chunk_id]
            if exclude_ids and chunk["parent_id"] in exclude_ids:
                continue
            key = (chunk["parent_type"], chunk["parent_id"])
            best[key] = max(best.get(key, 0.0), score)
        ranked = sorted(best, key=best.get, reverse=True)[: self.settings.n_parent]
        return [
            RetrievedParent(parent_type, parent_id, self.gold.get_parent(parent_type, parent_id), best[(parent_type, parent_id)], "search")
            for parent_type, parent_id in ranked
        ]

    def expand(self, parents: list[RetrievedParent]) -> list[RetrievedParent]:
        return expand_parents(self.gold, parents, self.settings.related_max)

    # --- 전체 흐름 --------------------------------------------------------
    def retrieve(
        self,
        question: str,
        *,
        expand_related: bool = True,
        query: str | None = None,
        extra_filters: dict[str, Any] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> RetrievalResult:
        """query: 검색에 쓸 문장 (없으면 question). extra_filters: 질문에서 뽑은 조건에 더할 필터.
        exclude_ids: 결과에서 뺄 parent (채팅에서 '비슷한 유물'을 찾을 때 현재 유물 제외)."""
        stats_question = is_stats_question(question)
        relic_ids, unknown_labels = self.find_labels(question)
        if unknown_labels and not relic_ids:
            # "신수 999999" 처럼 번호 형식인데 데이터에 없으면 비슷한 번호를 찾지 않고 근거 없음으로 답한다
            return RetrievalResult(question, "label", {}, [], False, stats_question)
        if relic_ids:
            parents = [
                RetrievedParent("relic", rid, self.gold.relic_cards[rid], 1.0, "label") for rid in relic_ids
            ]
            parents = self.expand(parents) if expand_related else parents
            return RetrievalResult(question, "label", {}, parents, True, stats_question)

        search_text = query or question
        filters = {**extract_filters(question), **(extra_filters or {})}
        fused, best_vector, best_bm25 = self.hybrid_search(search_text, filters)
        if not fused and set(filters) - {"record_type"}:
            # 조건이 너무 좁아 아무것도 안 걸리면 기본 필터로 다시 찾는다
            filters = {k: v for k, v in filters.items() if k == "record_type"}
            fused, best_vector, best_bm25 = self.hybrid_search(search_text, filters)

        found = bool(fused) and (
            best_vector >= self.settings.min_vector_similarity or best_bm25 >= self.settings.min_bm25_score
        )
        parents = self.to_parents(fused, exclude_ids) if found else []
        found = found and bool(parents)
        if expand_related and parents:
            parents = self.expand(parents)
        return RetrievalResult(
            question,
            "search",
            filters,
            parents,
            found,
            stats_question,
            best_vector,
            best_bm25,
            sorted(fused.items(), key=lambda item: item[1], reverse=True),
        )
