"""Business logic for validating uploads and running artifact classification."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from ai.vision.inference import ArtifactPredictor
from backend.config import ApiSettings
from backend.schemas import (
    ClassificationCandidate,
    ClassificationFailure,
    ClassificationSuccess,
)


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def decode_image(contents: bytes, settings: ApiSettings) -> Image.Image:
    try:
        with Image.open(BytesIO(contents)) as uploaded:
            if uploaded.format not in ALLOWED_FORMATS:
                raise ApiError(
                    415,
                    "INVALID_IMAGE_TYPE",
                    "JPG 또는 PNG 이미지만 업로드할 수 있습니다.",
                )
            width, height = uploaded.size
            if width * height > settings.max_image_pixels:
                raise ApiError(
                    413,
                    "IMAGE_TOO_LARGE",
                    "이미지 해상도가 허용 범위를 초과했습니다.",
                )
            uploaded.load()
            return uploaded.convert("RGB")
    except ApiError:
        raise
    except (
        Image.DecompressionBombError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ApiError(
            400,
            "INVALID_IMAGE",
            "이미지 파일을 읽을 수 없습니다.",
        ) from error


async def classify_image(
    contents: bytes,
    predictor: ArtifactPredictor,
    settings: ApiSettings,
) -> ClassificationSuccess | ClassificationFailure:
    """Decode an uploaded image and run it through the vision predictor."""
    decoded = await run_in_threadpool(decode_image, contents, settings)
    top_k = min(settings.top_k, len(predictor.class_to_idx))
    try:
        predictions = await run_in_threadpool(predictor.predict_pil, decoded, top_k)
    finally:
        decoded.close()

    best = predictions[0]
    if best.confidence < settings.confidence_threshold:
        return ClassificationFailure(
            message="유물을 인식하지 못했습니다. 다시 촬영해주세요."
        )
    return ClassificationSuccess(
        artifact_id=best.artifact_id,
        artifact_name=best.artifact_name,
        confidence=best.confidence,
        candidates=[
            ClassificationCandidate(
                artifact_id=prediction.artifact_id,
                confidence=prediction.confidence,
            )
            for prediction in predictions
        ],
    )
