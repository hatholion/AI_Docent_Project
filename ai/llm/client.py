"""YAML 설정만으로 Ollama 또는 Transformers 답변을 생성한다."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ai.llm.transformers_backend import generate_answer


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def generate_chat(messages: list[dict[str, str]], config: dict) -> str:
    if not config.get("enabled", True):
        raise RuntimeError("LLM이 비활성화되어 있습니다. llm.enabled: true로 설정하세요")
    provider = config["provider"]
    if provider == "transformers":
        answer = generate_answer(messages, config)
    elif provider == "ollama":
        model = ChatOllama(
            model=config["model"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            num_predict=config["num_predict"],
            num_ctx=config["num_ctx"],
            reasoning=config["reasoning"],
        )
        converted = []
        for message in messages:
            cls = SystemMessage if message["role"] == "system" else HumanMessage
            converted.append(cls(content=message["content"]))
        answer = _content_text(model.invoke(converted).content)
    else:
        raise ValueError(f"지원하지 않는 LLM provider: {provider}")
    if not answer:
        raise RuntimeError("LLM이 빈 답변을 반환했습니다")
    return answer
