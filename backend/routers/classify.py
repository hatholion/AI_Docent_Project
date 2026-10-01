"""Router for POST /api/v1/classify (docs/api_spec.md #1)."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile

from backend.schemas import ClassificationFailure, ClassificationSuccess
from backend.services.classification_service import (
    ALLOWED_CONTENT_TYPES,
    ApiError,
    classify_image,
)


router = APIRouter()


@router.post(
    "/api/v1/classify",
    response_model=ClassificationSuccess | ClassificationFailure,
    responses={
        400: {"description": "Invalid image"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported image type"},
        422: {"description": "Invalid request"},
    },
)
async def classify(
    request: Request,
    image: UploadFile,
) -> ClassificationSuccess | ClassificationFailure:
    settings = request.app.state.settings

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        await image.close()
        raise ApiError(
            415,
            "INVALID_IMAGE_TYPE",
            "JPG 또는 PNG 이미지만 업로드할 수 있습니다.",
        )

    contents = await image.read(settings.max_image_bytes + 1)
    await image.close()
    if len(contents) > settings.max_image_bytes:
        raise ApiError(
            413,
            "IMAGE_TOO_LARGE",
            "이미지 파일이 허용 용량을 초과했습니다.",
        )
    if not contents:
        raise ApiError(400, "INVALID_IMAGE", "빈 이미지 파일입니다.")

    predictor = request.app.state.predictor
    return await classify_image(contents, predictor, settings)
