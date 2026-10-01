"""Shared configuration for the artifact image classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VisionConfig:
    data_root: Path = PROJECT_ROOT / "data" / "processed"
    metadata_path: Path = PROJECT_ROOT / "data" / "metadata.csv"
    image_size: int = 224
    resize_size: int = 256
    batch_size: int = 16
    num_workers: int = 0
    seed: int = 42
