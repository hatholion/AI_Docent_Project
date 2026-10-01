"""Prompt templates for visitor-type-adapted artifact descriptions.

Grounds the LLM in the GOLD context_text so it does not invent facts about
artifacts it was not trained on in detail.
"""

from __future__ import annotations

GROUNDING_INSTRUCTION = (
    "당신은 국립중앙박물관 AI 도슨트입니다. 아래 [유물 정보]에 있는 사실만 사용해서 "
    "답변하세요. [유물 정보]에 없는 내용은 절대로 지어내지 말고, 확실하지 않은 내용은 "
    "언급하지 마세요."
)

VISITOR_INSTRUCTIONS: dict[str, str] = {
    "child": (
        "이제 이 유물을 초등학생 관람객에게 설명해주세요. 어려운 한자어는 풀어서 쓰고, "
        "쉬운 단어로 3문장 이내로 짧게 설명하세요."
    ),
    "general": (
        "이제 이 유물을 처음 관람하는 성인 관람객에게 설명해주세요. 친절하고 이해하기 "
        "쉬운 말투로 4~6문장으로 설명하세요."
    ),
    "expert": (
        "이제 이 유물을 연구자·전문가 관람객에게 설명해주세요. 재질, 시대, 크기, "
        "지정번호 등 구체적인 정보를 포함해 학술적인 어투로 설명하세요."
    ),
}


def build_prompt(context_text: str, visitor_type: str) -> str:
    if visitor_type not in VISITOR_INSTRUCTIONS:
        raise ValueError(f"Unknown visitor_type: {visitor_type}")

    return (
        f"{GROUNDING_INSTRUCTION}\n\n"
        f"[유물 정보]\n{context_text}\n\n"
        f"{VISITOR_INSTRUCTIONS[visitor_type]}"
    )


CHAT_TONE_INSTRUCTIONS: dict[str, str] = {
    "child": "초등학생이 이해할 수 있는 쉬운 말로, 2~4문장으로 답하세요.",
    "general": "친절하고 이해하기 쉬운 말투로, 3~5문장으로 답하세요.",
    "expert": "필요하면 재질·시대·지정번호 등 구체적 정보를 포함해 학술적인 어투로 답하세요.",
}


def build_chat_prompt(
    context_text: str,
    visitor_type: str,
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """history: [(role, content), ...] oldest first, role은 'user' 또는 'assistant'."""
    if visitor_type not in CHAT_TONE_INSTRUCTIONS:
        raise ValueError(f"Unknown visitor_type: {visitor_type}")

    parts = [
        GROUNDING_INSTRUCTION,
        "",
        f"[유물 정보]\n{context_text}",
    ]

    if history:
        history_lines = "\n".join(
            f"{'관람객' if role == 'user' else '도슨트'}: {content}" for role, content in history
        )
        parts.append(f"\n[이전 대화]\n{history_lines}")

    parts.append(
        f"\n{CHAT_TONE_INSTRUCTIONS[visitor_type]}\n\n관람객 질문: {question}"
    )
    return "\n".join(parts)
