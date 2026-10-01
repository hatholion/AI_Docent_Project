"""Smoke-test the FastAPI classification contract with a local checkpoint."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

from starlette.exceptions import StarletteDeprecationWarning


warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
    category=StarletteDeprecationWarning,
)

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.vision.config import PROJECT_ROOT
from backend.config import ApiSettings
from backend.main import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ApiSettings(
        model_path=args.checkpoint,
        metadata_path=PROJECT_ROOT / "data" / "metadata.csv",
        device=args.device,
        confidence_threshold=0.0,
    )
    application = create_app(settings=settings)
    image_bytes = args.image.read_bytes()
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/classify",
            files={"image": (args.image.name, image_bytes, "image/png")},
        )
        response.raise_for_status()
        payload = response.json()
        assert payload["artifact_id"]
        assert payload["artifact_name"]
        assert 0 <= payload["confidence"] <= 1
        assert len(payload["candidates"]) == 3

        invalid_type = client.post(
            "/api/v1/classify",
            files={"image": ("artifact.txt", b"not an image", "text/plain")},
        )
        assert invalid_type.status_code == 415
        assert invalid_type.json()["error"]["code"] == "INVALID_IMAGE_TYPE"

        corrupted = client.post(
            "/api/v1/classify",
            files={"image": ("broken.png", b"not an image", "image/png")},
        )
        assert corrupted.status_code == 400
        assert corrupted.json()["error"]["code"] == "INVALID_IMAGE"

        missing = client.post("/api/v1/classify")
        assert missing.status_code == 422
        assert missing.json()["error"]["code"] == "INVALID_REQUEST"

        preflight = client.options(
            "/api/v1/classify",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

        loaded_predictor = application.state.predictor
        low_confidence_app = create_app(
            settings=replace(settings, confidence_threshold=1.0),
            predictor=loaded_predictor,
        )
        with TestClient(low_confidence_app) as low_confidence_client:
            low_confidence = low_confidence_client.post(
                "/api/v1/classify",
                files={"image": (args.image.name, image_bytes, "image/png")},
            )
        assert low_confidence.status_code == 200
        assert low_confidence.json()["artifact_id"] is None
        assert low_confidence.json()["confidence"] == 0.0
        assert low_confidence.json()["candidates"] == []

        size_limit_app = create_app(
            settings=replace(settings, max_image_bytes=8),
            predictor=loaded_predictor,
        )
        with TestClient(size_limit_app) as size_limit_client:
            too_large = size_limit_client.post(
                "/api/v1/classify",
                files={"image": (args.image.name, image_bytes, "image/png")},
            )
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "IMAGE_TOO_LARGE"

    print(f"classification_status={response.status_code}")
    print(f"artifact_id={payload['artifact_id']}")
    print(f"confidence={payload['confidence']:.6f}")
    print("invalid_type_status=415")
    print("corrupted_image_status=400")
    print("missing_image_status=422")
    print("low_confidence_status=200")
    print("too_large_status=413")
    print("cors_preflight_status=200")
    print("classify_api_smoke_test=ok")


if __name__ == "__main__":
    main()
