"""외부 LLM 없이 실제 Gold/Chroma와 4개 멀티턴 연결 시나리오를 검증한다."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.llm.chat import make_service
from ai.rag.config import load_config


REWRITES = {
    "이 유물은 어디에서 발견됐어?": "서울 암사동 집터유적에서 출토된 빗살무늬토기",
    "그곳에서는 또 어떤 유물이 발견됐어?": "서울 암사동 집터유적에서 발견된 다른 유물",
    "비슷한 유물 두 개 알려줘.": "백자 달항아리와 비슷한 다른 유물 두 개",
    "그중 첫 번째는 어느 시대야?": "접수 702 백자 달항아리의 시대",
    "이 유물과 농경문 청동기를 비교해줘.": "본관 2789 금동 반가사유상과 신수 1794 농경문 청동기 비교",
    "둘 중 어느 것이 더 오래됐어?": "본관 2789 금동 반가사유상과 신수 1794 농경문 청동기의 시대 비교",
    "이 유물과 같은 재질의 다른 유물은?": "접수 702 백자 달항아리와 같은 백자 재질의 다른 유물",
    "그중 시대도 가장 비슷한 건 뭐야?": "접수 702와 같은 조선시대 백자 유물 중 이전에 제시한 유물",
}

ANSWERS = {
    "이 유물은 어디에서 발견됐어?": "서울 암사동 집터유적에서 출토되었습니다.",
    "비슷한 유물 두 개 알려줘.": "첫 번째는 접수 702, 두 번째는 건희 1601입니다.",
    "이 유물과 농경문 청동기를 비교해줘.": "본관 2789와 신수 1794를 비교했습니다.",
    "이 유물과 같은 재질의 다른 유물은?": "첫 번째는 건희 1601, 두 번째는 신수 3657입니다.",
}


def scripted_llm(messages: list[dict[str, str]], _config: dict) -> str:
    prompt = messages[-1]["content"]
    if "검색 질의 재작성기" in messages[0]["content"]:
        question = prompt.rsplit("현재 질문: ", 1)[-1].strip()
        rewritten = REWRITES[question]
        if question == "그곳에서는 또 어떤 유물이 발견됐어?" and "서울 암사동" not in prompt:
            raise AssertionError("Scenario 1 history가 Query Rewrite prompt에 없습니다")
        if question == "그중 첫 번째는 어느 시대야?" and "첫 번째는 접수 702" not in prompt:
            raise AssertionError("Scenario 2 첫 번째 대상이 history에 없습니다")
        if question == "둘 중 어느 것이 더 오래됐어?" and "본관 2789와 신수 1794" not in prompt:
            raise AssertionError("Scenario 3 비교 대상이 history에 없습니다")
        if question == "그중 시대도 가장 비슷한 건 뭐야?" and "건희 1601" not in prompt:
            raise AssertionError("Scenario 4 이전 검색 답변이 history에 없습니다")
        return rewritten
    match = re.search(r"현재 사용자 질문: (.+)\Z", prompt, re.DOTALL)
    if match:
        question = match.group(1).strip()
        return ANSWERS.get(question, "제공된 Gold와 검색 자료 범위에서 후속 답변을 생성했습니다.")
    return "현재 유물 Gold 자료를 바탕으로 최초 설명을 생성했습니다."


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config("ai/llm/configs/base.yaml")
    service = make_service(config)
    service.generate_fn = scripted_llm
    scenarios = [
        ("scenario_1", "신수 22891", "이 유물은 어디에서 발견됐어?", "그곳에서는 또 어떤 유물이 발견됐어?"),
        ("scenario_2", "접수 702", "비슷한 유물 두 개 알려줘.", "그중 첫 번째는 어느 시대야?"),
        ("scenario_3", "본관 2789", "이 유물과 농경문 청동기를 비교해줘.", "둘 중 어느 것이 더 오래됐어?"),
        ("scenario_4", "접수 702", "이 유물과 같은 재질의 다른 유물은?", "그중 시대도 가장 비슷한 건 뭐야?"),
    ]
    output = []
    for name, label, first, second in scenarios:
        initial = service.describe_relic(relic_label=label)
        first_result = service.answer_follow_up(first, session_id=initial["session_id"])
        second_result = service.answer_follow_up(second, session_id=initial["session_id"])
        session = service.get_session(initial["session_id"])
        if len(session["history"]) != 5:
            raise AssertionError(f"{name}: history 저장 개수가 올바르지 않습니다")
        output.append({
            "scenario": name,
            "current_relic_label": label,
            "first_retrieval_query": first_result["retrieval_query"],
            "second_retrieval_query": second_result["retrieval_query"],
            "second_retrieved_chunk_ids": [item["chunk_id"] for item in second_result["retrieved"]],
            "history_messages": len(session["history"]),
        })
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
