"""Gold 직접 조회와 전체 collection 검색을 결합한 멀티턴 도슨트."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

from ai.llm.client import generate_chat
from ai.llm.prompts import (
    DOCENT_SYSTEM_PROMPT,
    FIRST_FOLLOW_UP_INSTRUCTION,
    INITIAL_DESCRIPTION_INSTRUCTION,
    LATER_FOLLOW_UP_INSTRUCTION,
    REWRITE_SYSTEM_PROMPT,
    VISITOR_INSTRUCTIONS,
)
from ai.rag.document_builder import build_document_text
from ai.rag.gold_loader import get_relic_records_by_label, require_relic_label
from ai.rag.retriever import retrieve
from ai.rag.session import InMemorySessionStore, SessionNotFoundError

GenerateFn = Callable[[list[dict[str, str]], dict], str]
CONTEXT_REFERENCES = ("이 유물", "그 유물", "그곳", "그중", "그 중", "첫 번째", "둘 중", "아까")
LEADING_GREETING_PATTERNS = (
    re.compile(r"^\s*\*{0,2}(?:안녕하세요|안녕)[!！?？.,。~～\s]*\*{0,2}\s*"),
    re.compile(
        r"^\s*\*{0,2}질문(?:을\s*)?(?:해\s*)?주셔서\s*(?:정말\s*)?"
        r"(?:고마워요|감사해요|감사합니다)[!！?？.,。~～\s]*\*{0,2}\s*"
    ),
)


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(최근 대화 없음)"
    names = {"user": "사용자", "assistant": "도슨트"}
    return "\n".join(f"{names[item['role']]}: {item['content']}" for item in history)


def _format_gold(records: list[dict]) -> str:
    return "\n\n".join(
        f"[Gold chunk_id: {record['chunk_id']}]\n{build_document_text(record)}"
        for record in records
    )


def _format_retrieval(results: list[dict]) -> str:
    if not results:
        return "(관련 검색 자료 없음)"
    return "\n\n".join(
        f"[검색 chunk_id: {item['chunk_id']} | "
        f"소장품번호: {item['relic_label'] or item['metadata'].get('parent_id')}]\n{item['content']}"
        for item in results
    )


def _relic_name(records: list[dict]) -> str:
    """Gold text 첫 머리의 실제 유물명을 query context로 사용한다."""
    return records[0]["text"].split("—", 1)[0].strip()


def _without_leading_greeting(answer: str) -> str:
    """모델이 반복 생성한 답변 앞부분의 인사와 감사 표현만 제거한다."""
    cleaned = answer
    for pattern in LEADING_GREETING_PATTERNS:
        cleaned = pattern.sub("", cleaned, count=1)
    cleaned = cleaned.lstrip()
    return cleaned or answer


class DocentChatService:
    def __init__(self, *, config: dict, store: Any,
                 sessions: InMemorySessionStore | None = None,
                 generate_fn: GenerateFn = generate_chat):
        self.config = config
        self.store = store
        self.sessions = sessions or InMemorySessionStore()
        self.generate_fn = generate_fn
        self.gold_path = Path(config["data"]["gold_path"])

    def _visitor_instruction(self, visitor_type: str | None) -> tuple[str, str]:
        selected = visitor_type or self.config["llm"]["visitor_type"]
        allowed = self.config["llm"]["visitor_types"]
        if selected not in allowed or selected not in VISITOR_INSTRUCTIONS:
            raise ValueError("visitor_type은 child, general, expert 중 하나여야 합니다")
        instruction = DOCENT_SYSTEM_PROMPT
        instruction += "\n" + VISITOR_INSTRUCTIONS[selected]
        return selected, instruction

    def _resolve_label(self, relic_label: str | None) -> str:
        selected = relic_label if relic_label is not None else self.config["runtime"].get("initial_relic_label")
        return require_relic_label(selected)

    def describe_relic(self, *, relic_label: str | None = None,
                       session_id: str | None = None,
                       visitor_type: str | None = None) -> dict:
        """최초 설명: Retriever 없이 relic_label로 Gold를 직접 조회한다."""
        label = self._resolve_label(relic_label)
        records = get_relic_records_by_label(self.gold_path, label)
        relic_name = _relic_name(records)
        selected_visitor, instruction = self._visitor_instruction(visitor_type)
        instruction += "\n" + INITIAL_DESCRIPTION_INSTRUCTION
        prompt = (
            f"다음은 현재 유물 '{relic_name}'의 Gold 자료입니다. "
            f"'{label}'은 유물명이 아니라 소장품번호입니다. 이 유물의 핵심 특징을 설명하세요.\n\n"
            f"{_format_gold(records)}"
        )
        answer = self.generate_fn([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ], self.config["llm"])
        answer = _without_leading_greeting(answer)
        session = self.sessions.create(session_id=session_id, current_relic_label=label)
        self.sessions.set_current_relic(session.session_id, label)
        self.sessions.append(session.session_id, "assistant", answer)
        return {
            "session_id": session.session_id,
            "current_relic_label": label,
            "visitor_type": selected_visitor,
            "answer": answer,
            "source_chunk_ids": [record["chunk_id"] for record in records],
            "retrieval_used": False,
        }

    def rewrite_query(self, *, question: str, current_relic_label: str,
                      current_relic_name: str,
                      history: list[dict[str, str]]) -> str:
        if not self.config["query_rewrite"]["enabled"]:
            return question.strip()
        comparison = re.search(r"(?:이 유물|그 유물)과\s+(.+?)(?:을|를)\s*비교", question)
        if comparison:
            # 현재 유물은 direct context에 이미 있으므로 다른 비교 대상만 검색한다.
            return comparison.group(1).strip()
        if not any(reference in question for reference in CONTEXT_REFERENCES):
            return question.strip()
        prompt = (
            f"현재 유물: {current_relic_name} ({current_relic_label})\n\n"
            f"최근 대화:\n{_format_history(history)}\n\n"
            f"현재 질문: {question.strip()}"
        )
        rewritten = self.generate_fn([
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ], self.config["llm"]).strip()
        unresolved = any(reference in rewritten for reference in CONTEXT_REFERENCES)
        too_short = len(rewritten) < max(8, len(question.strip()) // 2)
        if rewritten and not unresolved and not too_short:
            return rewritten
        # 작은 모델이 핵심 명사를 버리거나 지시어를 남기는 경우 검색 recall을 보호한다.
        explicit = question.strip()
        for reference in ("이 유물", "그 유물"):
            explicit = explicit.replace(reference, f"{current_relic_name} ({current_relic_label})")
        if not any(reference in explicit for reference in CONTEXT_REFERENCES):
            return explicit
        # 그곳/그중/둘 중처럼 history 없이는 풀 수 없는 지시어만 최근 답변과 함께 보낸다.
        recent_messages = history[-2:]
        recent = " ".join(item["content"][:400] for item in recent_messages)
        return (
            f"현재 유물: {current_relic_name} ({current_relic_label}). "
            f"최근 대화 핵심: {recent}. 질문: {explicit}"
        )

    def answer_follow_up(self, question: str, *, session_id: str | None = None,
                         relic_label: str | None = None,
                         visitor_type: str | None = None,
                         top_k: int | None = None) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question은 비어 있지 않은 문자열이어야 합니다")
        if session_id is None:
            label = self._resolve_label(relic_label)
            get_relic_records_by_label(self.gold_path, label)
            session = self.sessions.create(current_relic_label=label)
        else:
            try:
                session = self.sessions.get(session_id)
            except SessionNotFoundError:
                if relic_label is None:
                    raise
                label = self._resolve_label(relic_label)
                get_relic_records_by_label(self.gold_path, label)
                session = self.sessions.create(session_id=session_id, current_relic_label=label)
            if relic_label is not None:
                label = self._resolve_label(relic_label)
                get_relic_records_by_label(self.gold_path, label)
                self.sessions.set_current_relic(session.session_id, label)
                session.current_relic_label = label
        label = self._resolve_label(session.current_relic_label)
        records = get_relic_records_by_label(self.gold_path, label)
        turns = self.config["chat"]["history_turns"]
        history = self.sessions.recent_history(session.session_id, turns)
        is_first_follow_up = not any(item["role"] == "user" for item in session.history)
        retrieval_query = self.rewrite_query(
            question=question,
            current_relic_label=label,
            current_relic_name=_relic_name(records),
            history=history,
        )
        results = retrieve(self.store, retrieval_query, self.config["retrieval"], top_k=top_k)
        current_chunk_ids = {record["chunk_id"] for record in records}
        retrieved_context = [item for item in results if item["chunk_id"] not in current_chunk_ids]
        selected_visitor, instruction = self._visitor_instruction(visitor_type)
        instruction += "\n" + (
            FIRST_FOLLOW_UP_INSTRUCTION if is_first_follow_up else LATER_FOLLOW_UP_INSTRUCTION
        )
        prompt = (
            f"현재 기준 유물: {_relic_name(records)} (소장품번호: {label})\n"
            f"현재 유물 Gold 자료:\n{_format_gold(records)}\n\n"
            f"전체 유물 DB 검색 자료:\n{_format_retrieval(retrieved_context)}\n\n"
            f"최근 대화:\n{_format_history(history)}\n\n"
            f"현재 사용자 질문: {question.strip()}"
        )
        answer = self.generate_fn([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ], self.config["llm"])
        if not is_first_follow_up:
            answer = _without_leading_greeting(answer)
        self.sessions.append(session.session_id, "user", question)
        self.sessions.append(session.session_id, "assistant", answer)
        return {
            "session_id": session.session_id,
            "current_relic_label": label,
            "visitor_type": selected_visitor,
            "question": question.strip(),
            "retrieval_query": retrieval_query,
            "answer": answer,
            "retrieved": results,
            "retrieval_used": True,
        }

    def get_session(self, session_id: str) -> dict:
        session = self.sessions.get(session_id)
        return {
            "session_id": session.session_id,
            "current_relic_label": session.current_relic_label,
            "history": session.history,
        }
