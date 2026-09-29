"""FastAPI application entrypoint: app creation, middleware, error handling, routers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ai.vision.inference import ArtifactPredictor
from backend.config import ApiSettings
from backend.db.artifacts import init_artifacts
from backend.db.conversations import setup_conversations
from backend.db.database import get_connection
from backend.routers.artifacts import router as artifacts_router
from backend.routers.classify import router as classify_router
from backend.services.classification_service import ApiError


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

        connection = await run_in_threadpool(get_connection)
        await run_in_threadpool(init_artifacts, connection)
        await run_in_threadpool(setup_conversations)
        application.state.db_connection = connection

        yield

        application.state.predictor = None
        connection.close()

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

    application.include_router(classify_router)
    application.include_router(artifacts_router)

    return application


app = create_app()
