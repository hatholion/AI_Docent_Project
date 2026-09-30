"""답변 생성 (기획서 §7): 카드 context_text만 근거로 답하고, 출처 번호를 후처리로 검사한다."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from ai.rag import ollama
from ai.rag.config import LLM_MODEL, TOTAL_COLLECTION_SIZE, GenerationSettings
from ai.rag.data import normalize_label
from ai.rag.retrieval import LABEL_RE, RetrievalResult, RetrievedParent


VisitorType = Literal["child", "general", "expert"]

NOT_FOUND_ANSWER = "자료에서 찾지 못했어요. 소장품 이름이나 번호(예: 신수 1846)로 다시 물어봐 주세요."

STYLE: dict[str, str] = {
    "child": "어린이가 이해할 수 있게 쉬운 낱말과 짧은 문장으로 친근하게 설명한다.",
    "general": "일반 관람객이 이해하기 쉽게 친절하게 설명한다.",
    "expert": "전문가에게 설명하듯 정확한 용어로 자세하게 설명한다.",
}

SYSTEM_PROMPT = """너는 국립중앙박물관 소장품 해설 도우미 '민속이'다. 반드시 아래 규칙을 지킨다.
1. [자료]에 적힌 내용으로만 답한다. 자료에 없는 사실(연대, 제작자, 의미, 다른 유물과의 비교 등)은 추측하지 않고, 일반 상식이나 시대 배경도 덧붙이지 않는다. 답할 근거가 없으면 "자료에서 찾지 못했어요"라고만 말한다.
2. "박물관 해설 없음"이라고 적힌 소장품은 분류 정보(국적·시대·재질·용도·출토지·크기)만 말하고, 그 이상은 말하지 않는다.
3. "자료 유형: 유리건판 사진"인 자료는 유물이 아니라 "○○를 촬영한 사진 자료"라고 소개한다.
4. 사실을 말한 문장 끝에 근거 소장품 번호를 {example}처럼 대괄호로 붙인다. 번호는 [자료] 첫 줄에 적힌 그대로(앞의 구분 이름까지) 옮겨 쓰고, [자료]에 없는 번호는 절대 쓰지 않는다.
5. 전체 소장품의 수량이나 통계는 답하지 않는다. [자료]는 전체 소장품의 일부 표본이다.
6. 같은 이름의 소장품이 여러 건이면 후보를 번호와 함께 나열한다.
7. 인사말과 맺음말 없이 바로 답한다. 출처 목록은 쓰지 않는다. 답변 끝에 시스템이 붙인다.
8. {style}""".strip()

SCOPE_NOTE = f"※ 이 자료는 국립중앙박물관 전체 소장품 {TOTAL_COLLECTION_SIZE:,}건 중 약 2.3%만 담고 있어 전체 수량·통계는 알려드릴 수 없어요."
BRACKET_RE = re.compile(r"\s?\[([^\[\]\n]{1,40})\]")
# Ollama로 EXAONE을 돌리면 답변 끝에 역할 이름이 새어 나오는 경우가 있다
TRAILING_ROLE_RE = re.compile(r"\s*(시스템|system|assistant|user)\s*$", re.I)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[str]
    thumbnails: list[dict[str, str]]
    cited_labels: list[str]
    removed_labels: list[str] = field(default_factory=list)
    route: str = "search"
    found: bool = True
    llm_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def build_context(parents: list[RetrievedParent]) -> str:
    return "\n\n".join(f"[자료 {i}]\n{p.card['context_text'].strip()}" for i, p in enumerate(parents, 1))


def allowed_labels(parents: list[RetrievedParent]) -> dict[str, RetrievedParent]:
    """답변에 쓸 수 있는 번호 → 근거 카드 (묶음·세트는 구성품 번호도 허용)."""
    allowed: dict[str, RetrievedParent] = {}
    for parent in parents:
        if parent.label:
            allowed.setdefault(normalize_label(parent.label), parent)
        for member in parent.card.get("member_labels", []):
            allowed.setdefault(normalize_label(member), parent)
    return allowed


def clean_citations(text: str, parents: list[RetrievedParent]) -> tuple[str, list[str], list[str]]:
    """넘기지 않은 카드의 [번호]를 지운다. (정리된 답변, 인용 번호, 지운 번호)"""
    allowed = allowed_labels(parents)
    cited: list[str] = []
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        labels = [part.strip() for part in re.split(r"[,·/]", match.group(1))]
        if not all(LABEL_RE.fullmatch(label) for label in labels):
            return match.group(0)  # 번호가 아닌 대괄호는 그대로
        kept = [label for label in labels if normalize_label(label) in allowed]
        removed.extend(label for label in labels if normalize_label(label) not in allowed)
        for label in kept:
            if label not in cited:
                cited.append(label)
        return f" [{', '.join(kept)}]" if kept else ""

    return BRACKET_RE.sub(replace, text), cited, removed


def collect_sources(cited: list[str], parents: list[RetrievedParent]) -> tuple[list[str], list[RetrievedParent]]:
    allowed = allowed_labels(parents)
    used: list[RetrievedParent] = []
    for label in cited:
        parent = allowed[normalize_label(label)]
        if parent not in used:
            used.append(parent)
    if not used and parents:
        # 인용이 없으면 LLM에 넘긴 주 근거 카드를 출처로 둔다 (1순위 카드 하나만 두면 답변과 출처가 어긋날 수 있다)
        used = [p for p in parents if p.via != "related"]
    return list(dict.fromkeys(p.citation for p in used)), used


def tidy(text: str) -> str:
    text = TRAILING_ROLE_RE.sub("", text.strip())
    return text.replace("本관", "본관")


def tidy_after_citations(text: str) -> str:
    # 굵게 표시 안의 번호를 지우면 남는 빈 '****' 정리
    return re.sub(r"\*\*\s*\*\*\s?", "", text)


def photo_plate_lead(parents: list[RetrievedParent], text: str) -> str:
    """유리건판 사진은 유물이 아니라 사진 자료로 소개한다 (기능 요구사항 3). 모델이 놓치면 첫 문장을 붙인다."""
    plate = next((p for p in parents if p.card.get("record_type") == "photo_plate"), None)
    if plate is None or "사진" in text:
        return text
    lead = f"[{plate.label}]은 유물이 아니라 「{plate.card['title']}」을(를) 촬영한 유리건판 사진 자료예요."
    return f"{lead}\n\n{text}"


def stats_answer(retrieval: RetrievalResult) -> Answer:
    """수량·통계 질문은 LLM이 예시를 세어 '총 N점'처럼 답하지 않도록 고정 안내 + 관련 예시 목록으로 답한다."""
    lines = [SCOPE_NOTE, "", "대신 자료에서 찾은 관련 소장품을 소개할게요."]
    for parent in retrieval.parents[:5]:
        if parent.parent_type == "relic":
            lines.append(f"- {parent.card['title']} [{parent.label}]")
        else:
            lines.append(f"- {parent.card['title']} ({parent.card.get('member_count', 0)}점 묶음)")
    cited = [p.label for p in retrieval.parents[:5] if p.label]
    sources, used = collect_sources(cited, retrieval.parents[:5])
    return Answer(retrieval.question, "\n".join(lines), sources, [], cited, route=retrieval.route, found=True)


EXPLORE_LEAD: dict[str, str] = {
    "child": "「{title}」과(와) 비슷한 소장품을 찾아봤어요!",
    "general": "「{title}」과(와) 관련된 소장품을 자료에서 찾았어요.",
    "expert": "「{title}」 기준으로 자료에서 검색된 소장품입니다.",
}
RELATED_LEAD: dict[str, str] = {
    "child": "「{title}」의 연관 소장품이에요!",
    "general": "「{title}」의 연관 소장품이에요.",
    "expert": "「{title}」의 연관 소장품으로 기록된 자료입니다.",
}
EXPLORE_NOT_FOUND = "자료에서 비슷한 소장품을 찾지 못했어요. 시대나 재질을 넣어 다시 물어봐 주세요."
FILTER_NAMES = {
    "period": "시대",
    "material_l1": "재질",
    "material_l2": "재질",
    "designations": "지정",
    "find_place_l1": "출토지",
    "purpose_l1": "용도",
    "purpose_l2": "용도",
}


def explore_answer(retrieval: RetrievalResult, visitor_type: VisitorType) -> Answer:
    """탐색 질문("비슷한 유물", "같은 시대")은 LLM 없이 찾은 카드를 나열한다. 모든 줄이 넘긴 카드에서 나온다."""
    base = next((p for p in retrieval.parents if p.via == "artifact"), None)
    found = [p for p in retrieval.parents if p.via != "artifact"]
    if not found:
        return Answer(retrieval.question, EXPLORE_NOT_FOUND, [], [], [], route=retrieval.route, found=False)

    lead = RELATED_LEAD if all(p.via == "related" for p in found) else EXPLORE_LEAD
    lines = [lead[visitor_type].format(title=base.card["title"] if base else "현재 유물")]
    conditions = [f"{FILTER_NAMES[k]} {v}" for k, v in retrieval.filters.items() if k in FILTER_NAMES]
    if conditions:
        lines.append(f"(조건: {', '.join(conditions)})")
    lines.append("")
    for parent in found:
        if parent.parent_type == "relic":
            attributes = parent.card.get("attributes", {})
            facts = " · ".join(filter(None, [attributes.get("period"), attributes.get("material")]))
            lines.append(f"- {parent.card['title']} [{parent.label}]" + (f" — {facts}" if facts else ""))
        else:
            lines.append(f"- {parent.card['title']} ({parent.card.get('member_count', 0)}점 묶음)")

    cited = [p.label for p in found if p.label]
    sources = list(dict.fromkeys(p.citation for p in found))
    thumbnails = [
        {"label": p.label or "", "title": p.card["title"], "url": p.card["thumbnail"]}
        for p in found
        if p.card.get("thumbnail")
    ]
    return Answer(retrieval.question, "\n".join(lines), sources, thumbnails, cited, route=retrieval.route, found=True)


def generate_answer(
    retrieval: RetrievalResult,
    *,
    visitor_type: VisitorType = "general",
    settings: GenerationSettings | None = None,
    model: str = LLM_MODEL,
    history: list[dict[str, str]] | None = None,
) -> Answer:
    """history: 이전 대화 [{role: user|assistant, content}] (오래된 순). 채팅에서 앞 대화를 이어갈 때 쓴다."""
    settings = settings or GenerationSettings()
    if not retrieval.found or not retrieval.parents:
        return Answer(retrieval.question, NOT_FOUND_ANSWER, [], [], [], route=retrieval.route, found=False)

    if retrieval.stats_question:
        return stats_answer(retrieval)

    if retrieval.list_answer:
        return explore_answer(retrieval, visitor_type)

    # 예시 번호를 실제 자료의 번호로 보여줘야 모델이 구분 이름(신수·덕수 등)을 베끼지 않는다
    example = next((f"[{p.label}]" for p in retrieval.parents if p.label), "[소장품 번호]")
    system = SYSTEM_PROMPT.format(total=TOTAL_COLLECTION_SIZE, style=STYLE[visitor_type], example=example)
    user = f"[자료]\n{build_context(retrieval.parents)}\n\n[질문]\n{retrieval.question}"

    started = time.perf_counter()
    previous = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in (history or [])
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    raw = ollama.chat(
        [{"role": "system", "content": system}, *previous, {"role": "user", "content": user}],
        model=model,
        temperature=settings.temperature,
        num_ctx=settings.num_ctx,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
    )
    llm_seconds = time.perf_counter() - started

    text, cited, removed = clean_citations(tidy(raw), retrieval.parents)
    text = photo_plate_lead(retrieval.parents, tidy_after_citations(text))
    sources, used = collect_sources(cited, retrieval.parents)
    thumbnails = [
        {"label": p.label or "", "title": p.card["title"], "url": p.card["thumbnail"]}
        for p in used
        if p.card.get("thumbnail")
    ]
    return Answer(
        retrieval.question,
        text.strip(),
        sources,
        thumbnails,
        cited,
        removed,
        route=retrieval.route,
        found=True,
        llm_seconds=round(llm_seconds, 2),
    )
