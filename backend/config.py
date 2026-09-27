"""Environment-backed settings for the classification API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ai.vision.config import PROJECT_ROOT


@dataclass(frozen=True)
class ApiSettings:
    model_path: Path
    metadata_path: Path = PROJECT_ROOT / "data" / "metadata.csv"
    device: str = "auto"
    confidence_threshold: float = 0.60
    top_k: int = 3
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://localhost:3000",
    )

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported VISION_DEVICE: {self.device}")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("VISION_CONFIDENCE_THRESHOLD must be between 0 and 1")
        if self.top_k < 1:
            raise ValueError("VISION_TOP_K must be at least 1")
        if self.max_image_bytes < 1 or self.max_image_pixels < 1:
            raise ValueError("Image size limits must be positive")

    @classmethod
    def from_env(cls) -> "ApiSettings":
        default_model = PROJECT_ROOT / "runs" / "vision" / "fine_tune_gpu" / "best_model.pth"
        return cls(
            model_path=Path(os.getenv("VISION_MODEL_PATH", str(default_model))),
            metadata_path=Path(
                os.getenv(
                    "VISION_METADATA_PATH",
                    str(PROJECT_ROOT / "data" / "metadata.csv"),
                )
            ),
            device=os.getenv("VISION_DEVICE", "auto"),
            confidence_threshold=float(
                os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.60")
            ),
            top_k=int(os.getenv("VISION_TOP_K", "3")),
            max_image_bytes=int(
                os.getenv("VISION_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))
            ),
            max_image_pixels=int(
                os.getenv("VISION_MAX_IMAGE_PIXELS", "25000000")
            ),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "CORS_ALLOWED_ORIGINS",
                    "http://localhost:5173,http://localhost:3000",
                ).split(",")
                if origin.strip()
            ),
        )
