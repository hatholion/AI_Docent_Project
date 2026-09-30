"""Smoke-test /docent/description and /docent/chat end-to-end (needs Ollama running).

Uses a fake predictor (classify isn't under test here) but real DB + real GOLD data +
real local LLM call, so this exercises the actual RAG/LLM integration.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from ai.vision.inference import RankedPrediction
from backend.main import create_app


class FakePredictor:
    class_to_idx = {"bon002789": 0}

    def predict_pil(self, image, top_k: int = 3):
        return [RankedPrediction(artifact_id="bon002789", artifact_name="금동 반가사유상", confidence=0.95)]


def main() -> None:
    app = create_app(predictor=FakePredictor())
    with TestClient(app) as client:
        desc = client.post(
            "/api/v1/docent/description",
            json={"artifact_id": "bon002789", "visitor_type": "child"},
        )
        desc.raise_for_status()
        desc_body = desc.json()
        assert desc_body["artifact_id"] == "bon002789"
        assert desc_body["visitor_type"] == "child"
        assert len(desc_body["description"]) > 0
        print(f"description_status={desc.status_code} chars={len(desc_body['description'])}")

        chat1 = client.post(
            "/api/v1/docent/chat",
            json={
                "artifact_id": "bon002789",
                "visitor_type": "general",
                "question": "이 유물은 어느 시대 거예요?",
            },
        )
        chat1.raise_for_status()
        chat1_body = chat1.json()
        session_id = chat1_body["session_id"]
        assert session_id
        assert len(chat1_body["answer"]) > 0
        print(f"chat1_status={chat1.status_code} session_id={session_id}")
        print(f"chat1_answer={chat1_body['answer']}")

        chat2 = client.post(
            "/api/v1/docent/chat",
            json={
                "artifact_id": "bon002789",
                "visitor_type": "general",
                "session_id": session_id,
                "question": "방금 말한 그 시대에 다른 유명한 유물도 있어요?",
            },
        )
        chat2.raise_for_status()
        chat2_body = chat2.json()
        assert chat2_body["session_id"] == session_id
        print(f"chat2_status={chat2.status_code}")
        print(f"chat2_answer={chat2_body['answer']}")

        history = client.get(f"/api/v1/docent/chat/{session_id}")
        history.raise_for_status()
        history_body = history.json()
        assert len(history_body["messages"]) == 4
        print(f"history_status={history.status_code} messages={len(history_body['messages'])}")

        missing_artifact = client.post(
            "/api/v1/docent/description",
            json={"artifact_id": "nope0000", "visitor_type": "child"},
        )
        assert missing_artifact.status_code == 404
        print(f"missing_artifact_status={missing_artifact.status_code}")

        missing_session = client.get("/api/v1/docent/chat/sess_does_not_exist")
        assert missing_session.status_code == 404
        print(f"missing_session_status={missing_session.status_code}")

    print("docent_api_smoke_test=ok")


if __name__ == "__main__":
    main()
