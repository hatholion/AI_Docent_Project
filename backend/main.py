"""FastAPI application exposing the artifact classification endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


def create_app(
    settings: ApiSettings | None = None,
    predictor: ArtifactPredictor | None = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if predictor is not None:
            application.state.predictor = predictor
        else:
            application.state.predictor = await run_in_threadpool(
                ArtifactPredictor.from_paths,
                resolved_settings.model_path,
                resolved_settings.metadata_path,
                resolved_settings.device,
            )
        yield
        application.state.predictor = None

    application = FastAPI(
        title="국립중앙박물관 AI 도슨트 API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @application.exception_handler(ApiError)
    async def handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request,
        __: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "요청 형식이 올바르지 않습니다.",
                }
            },
        )

    @application.post(
        "/api/v1/classify",
        response_model=ClassificationSuccess | ClassificationFailure,
        responses={
            400: {"description": "Invalid image"},
            413: {"description": "Image too large"},
            415: {"description": "Unsupported image type"},
            422: {"description": "Invalid request"},
        },
    )
    async def classify(image: UploadFile) -> ClassificationSuccess | ClassificationFailure:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            await image.close()
            raise ApiError(
                415,
                "INVALID_IMAGE_TYPE",
                "JPG 또는 PNG 이미지만 업로드할 수 있습니다.",
            )

        contents = await image.read(resolved_settings.max_image_bytes + 1)
        await image.close()
        if len(contents) > resolved_settings.max_image_bytes:
            raise ApiError(
                413,
                "IMAGE_TOO_LARGE",
                "이미지 파일이 허용 용량을 초과했습니다.",
            )
        if not contents:
            raise ApiError(400, "INVALID_IMAGE", "빈 이미지 파일입니다.")

        decoded = await run_in_threadpool(decode_image, contents, resolved_settings)
        active_predictor: ArtifactPredictor = application.state.predictor
        top_k = min(resolved_settings.top_k, len(active_predictor.class_to_idx))
        try:
            predictions = await run_in_threadpool(
                active_predictor.predict_pil,
                decoded,
                top_k,
            )
        finally:
            decoded.close()
        best = predictions[0]
        if best.confidence < resolved_settings.confidence_threshold:
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

    return application


app = create_app()
