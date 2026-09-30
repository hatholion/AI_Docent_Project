"""체크포인트 없이 가짜 모델로 백엔드를 띄워 프론트-백엔드 연동만 확인하는 개발용 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from PIL import Image

from ai.vision.inference import RankedPrediction
from backend.main import create_app


class FakePredictor:
    """항상 같은 유물을 반환하는 테스트용 가짜 예측기."""

    class_to_idx = {"bon002789": 0}

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> list[RankedPrediction]:
        return [
            RankedPrediction(
                artifact_id="bon002789",
                artifact_name="금동 반가사유상",
                confidence=0.95,
            )
        ]


app = create_app(predictor=FakePredictor())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
