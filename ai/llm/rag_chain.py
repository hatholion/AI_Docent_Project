"""선택 유물의 RAG 컨텍스트로 Ollama 또는 Transformers 답변을 생성한다."""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from ai.rag.context import build_context
from ai.llm.transformers_backend import generate_answer

VISITOR_INSTRUCTIONS = {
    "test": "",
    "child": "어린이 관람객에게 쉬운 어휘와 짧은 문장으로 설명하세요. 어려운 전문 용어는 피하거나 쉽게 풀어주세요. 말투는 유치원 선생님처럼 해주세요.",
    "general": "일반 성인 관람객에게 박물관 도슨트 수준의 명확한 말로 설명하세요.",
    "expert": "전문적 배경지식이 있는 관람객에게 정확한 전문 용어, 제작 기법, 역사적·미술사적 맥락을 자료가 뒷받침하는 범위에서 깊이 있게 설명하세요.",
}
BASE_INSTRUCTION = "제공된 유물 자료만 근거로 한국어로 답하세요. 근거가 없으면 자료에서 확인할 수 없다고 말하세요. 근거 유물 ID를 표시하세요."


def ask_rag(store, question: str, retriever_config: dict, llm_config: dict,
            *, relic_id: str | None = None) -> dict:
    """visitor_type은 답변 표현에만 사용하고 검색에는 전달하지 않는다."""
    if not llm_config["enabled"]:
        raise ValueError("LLM이 활성화되지 않았습니다. llm.enabled: true로 설정하세요")
    provider = llm_config["provider"]
    if provider not in ("ollama", "transformers"):
        raise ValueError("llm.provider는 ollama 또는 transformers여야 합니다")
    visitor_type = llm_config.get("visitor_type")
    if visitor_type not in VISITOR_INSTRUCTIONS:
        raise ValueError("llm.visitor_type은 test, child, general, expert 중 하나여야 합니다")
    context, docs = build_context(store, question, relic_id, retriever_config)
    instruction = BASE_INSTRUCTION
    if VISITOR_INSTRUCTIONS[visitor_type]:
        instruction += " " + VISITOR_INSTRUCTIONS[visitor_type]
    prompt = f"자료:\n{context}\n\n질문: {question}"
    if provider == "transformers":
        answer = generate_answer([
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ], llm_config)
    else:
        model = ChatOllama(model=llm_config["model"], temperature=llm_config["temperature"])
        answer = model.invoke([SystemMessage(content=instruction), HumanMessage(content=prompt)]).content
    return {"answer": answer, "visitor_type": visitor_type,
            "relic_id": relic_id, "retrieved_ids": [doc.metadata["relic_id"] for doc in docs]}
