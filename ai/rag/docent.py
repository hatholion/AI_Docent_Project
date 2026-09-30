"""도슨트 서비스 진입점: 기본은 룰(ID 조회), RAG 검색은 채팅 보조.

- describe(): 인식된 유물 설명. 항상 ID로 카드를 조회한다 (검색 없음).
- chat(): 질문을 아래 순서로 라우팅한다.
    1. 통계 질문          → 고정 안내 + 예시 (search)
    2. 소장품 번호         → 번호 색인 조회 (label, 현재 유물 번호면 artifact)
    3. 다른 소장품 이름    → 이름 사전 조회 (name). 동명이 많으면 검색으로 넘김
                            ("이 ○○", 현재 유물 이름이 있으면 건너뜀)
    4. 탐색 표현          → RAG 검색 (search). "비슷한 유물", "같은 시대" 등, 현재 유물 기준
    5. 그 외 (기본)       → 현재 유물 카드 (artifact)

python -m ai.rag.docent bon002789 --describe
python -m ai.rag.docent bon002789 "비슷한 유물 또 있어?" --no-llm
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ai.rag.config import PROJECT_ROOT, RetrievalSettings
from ai.rag.data import Card, GoldData, normalize_label
from ai.rag.generate import Answer, VisitorType, generate_answer
from ai.rag.retrieval import (
    LABEL_RE,
    RetrievalResult,
    RetrievedParent,
    Retriever,
    expand_parents,
    is_stats_question,
)
from ai.rag.text import has_hanja, hanja_reading


METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"

# 현재 유물을 벗어나 다른 소장품을 찾는 표현 → RAG 검색
EXPLORE_RE = re.compile(
    r"비슷한|유사한|닮은|다른\s?(유물|소장품|것|작품)|또\s?(있|뭐|어떤)|더\s?(보여|알려|있)|추천|"
    r"같은\s?(시대|재질|종류|지역|곳|용도)|관련된\s?(유물|소장품)|찾아\s?(줘|봐|주)|보여\s?(줘|주)"
)
# "관련된 소장품", "연관 유물" → 검색 대신 카드에 기록된 연관 소장품 목록
# e뮤지엄의 연관 소장품은 함께 출토된 유물이 아니므로 "같이 출토된"은 넣지 않는다
RELATED_RE = re.compile(r"(관련된|관련\s?있는|연관된?)\s?(유물|소장품|것|작품)")
# "이 백자 항아리는…", "이거 언제…"처럼 지금 보고 있는 유물을 가리키는 표현. 다른 소장품 이름 조회를 건너뛴다.
THIS_RE = re.compile(r"^\s*(이|이거|이것|이게|이건|얘)\s|이거|이것|이게|이건|이\s?유물|이\s?작품|이\s?소장품")
# "같은 ○○" → 현재 유물의 metadata로 필터
SAME_FILTERS = {
    "시대": ("period",),
    "재질": ("material_l2", "material_l1"),
    "지역": ("find_place_l1",),
    "곳": ("find_place_l1",),
    "용도": ("purpose_l2", "purpose_l1"),
}
NAME_MIN_LEN = 4  # '토기', '구슬' 같은 짧은 이름은 일반 명사와 겹쳐서 이름 사전에 넣지 않는다
NAME_MAX_CANDIDATES = 3  # 이보다 동명이 많으면 이름 조회 대신 검색으로 넘긴다
PUNCT_RE = re.compile(r"[\s()（）《》「」『』\[\],.·ㆍ'\"]+")


def normalize_name(text: str) -> str:
    return PUNCT_RE.sub("", text).lower()


class ArtifactNotFound(LookupError):
    pass


@dataclass
class ChatRoute:
    route: str  # artifact / label / name / search / not_found
    reason: str
    retrieval: RetrievalResult


class Docent:
    def __init__(
        self,
        *,
        gold: GoldData | None = None,
        settings: RetrievalSettings | None = None,
        metadata_csv: Path = METADATA_CSV,
    ) -> None:
        self.gold = gold or GoldData()
        self.settings = settings or RetrievalSettings()
        self.metadata_csv = metadata_csv

    # --- 색인 없이 쓰는 룰 ----------------------------------------------
    @cached_property
    def retriever(self) -> Retriever:
        """RAG 검색이 필요할 때만 벡터·BM25 색인을 연다."""
        return Retriever(self.settings, gold=self.gold)

    @cached_property
    def artifact_labels(self) -> dict[str, str]:
        """metadata.csv artifact_id(Vision 클래스, 예: bon002789) → 소장품 번호(본관 2789)."""
        if not self.metadata_csv.is_file():
            return {}
        with self.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
            return {
                row["artifact_id"].strip(): row["accession_no"].strip()
                for row in csv.DictReader(handle)
                if row.get("accession_no")
            }

    @cached_property
    def name_index(self) -> dict[str, list[str]]:
        """정규화한 명칭·이명·한자 독음 → relic_id 목록 (유물만)."""
        index: dict[str, list[str]] = {}
        for relic_id, card in self.gold.relic_cards.items():
            if card["record_type"] != "artifact":
                continue
            for name in self._names(card):
                if len(name) >= NAME_MIN_LEN and relic_id not in index.setdefault(name, []):
                    index[name].append(relic_id)
        return index

    @staticmethod
    def _names(card: Card) -> set[str]:
        names = {card["title"], *card.get("alt_names", [])}
        names |= {hanja_reading(name) for name in list(names) if has_hanja(name)}
        # '금동약사여래입상(本館 325-1)' 처럼 괄호 설명이 붙은 경우 앞부분도 이름으로 본다
        names |= {re.split(r"[(（]", name)[0] for name in list(names)}
        return {normalize_name(name) for name in names if name}

    def resolve(self, ref: str) -> str:
        """artifact_id(bon002789) / 소장품 번호(본관 2789) / relic_id(nmm-bon-002789-00) → relic_id."""
        ref = ref.strip()
        if ref in self.gold.relic_cards:
            return ref
        label = self.artifact_labels.get(ref, ref)
        relic_id = self.gold.label_index.get(normalize_label(label))
        if relic_id is None:
            raise ArtifactNotFound(f"'{ref}'에 해당하는 소장품이 GOLD 데이터에 없습니다.")
        return relic_id

    def artifact_parents(self, relic_id: str, via: str = "artifact") -> list[RetrievedParent]:
        # ID로 정해진 유물은 연관 소장품 카드를 넘기지 않는다. 넘기면 모델이 연관 유물의 출토지·크기를
        # 현재 유물 것으로 답한다. 연관 소장품 이름·번호는 카드 context_text의 '연관 소장품' 줄에 이미 있다.
        card = self.gold.relic_cards[relic_id]
        return expand_parents(self.gold, [RetrievedParent("relic", relic_id, card, 1.0, via)], related_max=0)

    @cached_property
    def parent_metadata(self) -> dict[str, dict[str, Any]]:
        """relic_id → 필터용 metadata. 묶음에 속한 소장품은 자기 청크가 없어 카드 속성으로 채운다."""
        result = {c["parent_id"]: c["metadata"] for c in self.gold.chunks if c["parent_type"] == "relic"}
        for relic_id, card in self.gold.relic_cards.items():
            if relic_id not in result:
                attributes = card.get("attributes", {})
                material = (attributes.get("material") or "").split(" > ")
                result[relic_id] = {
                    "period": attributes.get("period"),
                    "material_l1": material[0] or None,
                    "material_l2": material[1] if len(material) > 1 else None,
                }
        return result

    @staticmethod
    def _result(question: str, route: str, parents: list[RetrievedParent]) -> RetrievalResult:
        return RetrievalResult(question, route, {}, parents, bool(parents), False)

    # --- 라우팅 ----------------------------------------------------------
    def route_chat(self, relic_id: str, question: str) -> ChatRoute:
        current = self.gold.relic_cards[relic_id]

        if is_stats_question(question):
            return ChatRoute("search", "통계 질문", self.retriever.retrieve(question))

        labels = [m.group(0) for m in LABEL_RE.finditer(question)]
        if labels:
            ids = [self.gold.label_index.get(normalize_label(label)) for label in labels]
            others = [rid for rid in ids if rid and rid != relic_id]
            if others:
                parents = [p for rid in others for p in self.artifact_parents(rid, "label")]
                return ChatRoute("label", f"소장품 번호 {labels}", self._result(question, "label", parents))
            if relic_id not in ids and any(len(m.group(1)) >= 2 for m in LABEL_RE.finditer(question)):
                return ChatRoute("not_found", f"없는 번호 {labels}", self._result(question, "label", []))

        normalized = normalize_name(question)
        current_names = self._names(current)
        about_current = THIS_RE.search(question) or any(
            name in normalized for name in current_names if len(name) >= NAME_MIN_LEN
        )
        if not about_current:
            matched = self._match_other_name(normalized, relic_id)
            if matched:
                name, ids = matched
                if len(ids) <= NAME_MAX_CANDIDATES:
                    parents = [p for rid in ids for p in self.artifact_parents(rid, "name")]
                    return ChatRoute("name", f"이름 '{name}'", self._result(question, "name", parents))
                result = self.retriever.retrieve(question)
                return ChatRoute("search", f"동명 {len(ids)}건 '{name}' → 검색", result)

        if EXPLORE_RE.search(question) or RELATED_RE.search(question):
            return ChatRoute("search", "탐색 표현", self._explore(relic_id, question))

        return ChatRoute("artifact", "현재 유물", self._result(question, "artifact", self.artifact_parents(relic_id)))

    def _match_other_name(self, normalized_question: str, relic_id: str) -> tuple[str, list[str]] | None:
        best: tuple[str, list[str]] | None = None
        for name, ids in self.name_index.items():
            if name in normalized_question and relic_id not in ids:
                if best is None or len(name) > len(best[0]):
                    best = (name, ids)
        return best

    def _explore(self, relic_id: str, question: str) -> RetrievalResult:
        """현재 유물을 기준으로 다른 소장품을 찾는다. 현재 유물 카드는 비교용으로 맨 앞에 둔다."""
        current = self.gold.relic_cards[relic_id]
        base = RetrievedParent("relic", relic_id, current, 1.0, "artifact")
        related = [
            RetrievedParent("relic", r["relic_id"], self.gold.relic_cards[r["relic_id"]], 1.0, "related")
            for r in current.get("related", [])
            if r.get("collected") and r["relic_id"] in self.gold.relic_cards
        ]
        if RELATED_RE.search(question) and related:
            result = self._result(question, "search", [base, *related[: self.settings.n_parent]])
            result.list_answer = True
            return result

        metadata = self.parent_metadata.get(relic_id, {})
        extra_filters: dict[str, Any] = {}
        for word, keys in SAME_FILTERS.items():
            if re.search(rf"같은\s?{word}", question):
                key = next((k for k in keys if metadata.get(k)), None)
                if key:
                    extra_filters[key] = metadata[key]
        exclude = {relic_id, *(filter(None, [current.get("set_id")]))}
        result = self.retriever.retrieve(
            question,
            query=f"{current['title']} {question}",
            extra_filters=extra_filters,
            exclude_ids=exclude,
            expand_related=False,
        )
        result.parents = [base, *result.parents]
        result.found = True
        result.list_answer = True
        return result

    # --- 서비스 ----------------------------------------------------------
    def describe(self, ref: str, *, visitor_type: VisitorType = "general") -> Answer:
        relic_id = self.resolve(ref)
        card = self.gold.relic_cards[relic_id]
        question = f"{card['title']}({card['relic_label']})을(를) 관람객에게 소개해 줘."
        return generate_answer(self._result(question, "artifact", self.artifact_parents(relic_id)), visitor_type=visitor_type)

    def chat(
        self,
        ref: str,
        question: str,
        *,
        visitor_type: VisitorType = "general",
        history: list[dict[str, str]] | None = None,
    ) -> tuple[Answer, ChatRoute]:
        relic_id = self.resolve(ref)
        route = self.route_chat(relic_id, question)
        answer = generate_answer(route.retrieval, visitor_type=visitor_type, history=history)
        answer.route = route.route
        return answer, route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", help="artifact_id(bon002789) / 소장품 번호 / relic_id")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--describe", action="store_true", help="유물 설명 (ID 조회)")
    parser.add_argument("--visitor-type", choices=("child", "general", "expert"), default="general")
    parser.add_argument("--no-llm", action="store_true", help="라우팅과 근거 카드만 출력")
    args = parser.parse_args()

    docent = Docent()
    relic_id = docent.resolve(args.artifact)
    if args.describe or not args.question:
        parents = docent.artifact_parents(relic_id)
        print(f"route=artifact (ID 조회) cards={[p.label or p.parent_id for p in parents]}")
        if not args.no_llm:
            answer = docent.describe(args.artifact, visitor_type=args.visitor_type)
            print(f"\n{answer.answer}\n\n" + "\n".join(f"출처: {s}" for s in answer.sources))
        return

    route = docent.route_chat(relic_id, args.question)
    print(f"route={route.route} ({route.reason}) filters={route.retrieval.filters}")
    for rank, parent in enumerate(route.retrieval.parents, 1):
        print(f"  {rank}. [{parent.via}] {parent.label or parent.parent_id} {parent.card['title']}")
    if not args.no_llm:
        answer = generate_answer(route.retrieval, visitor_type=args.visitor_type)
        print(f"\n{answer.answer}\n\n" + "\n".join(f"출처: {s}" for s in answer.sources))


if __name__ == "__main__":
    main()
