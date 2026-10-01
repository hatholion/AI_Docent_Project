"""Reusable single-image inference for the artifact classifier."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ai.vision.config import VisionConfig
from ai.vision.data import build_transforms
from ai.vision.model import create_efficientnet_b0
from ai.vision.train import select_device


@dataclass(frozen=True)
class RankedPrediction:
    artifact_id: str
    artifact_name: str
    confidence: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_name": self.artifact_name,
            "confidence": self.confidence,
        }


class ArtifactPredictor:
    """Load one checkpoint and reuse it for PIL image predictions."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        metadata_path: Path,
        device: torch.device,
    ) -> None:
        self.checkpoint_path = checkpoint_path.resolve()
        self.metadata_path = metadata_path.resolve()
        self.device = device
        self.artifact_names = self._load_artifact_names(self.metadata_path)
        checkpoint = self._load_checkpoint(self.checkpoint_path)
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_stage = checkpoint.get("training_stage", "classifier_head")
        self.class_to_idx = self._validate_class_mapping(checkpoint.get("class_to_idx"))
        self.idx_to_class = {
            class_index: artifact_id
            for artifact_id, class_index in self.class_to_idx.items()
        }

        self.model = create_efficientnet_b0(
            len(self.class_to_idx),
            pretrained=False,
            freeze_backbone=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        config = replace(
            VisionConfig(),
            image_size=int(checkpoint.get("image_size", 224)),
            resize_size=int(checkpoint.get("resize_size", 256)),
        )
        self.transform = build_transforms(config)["test"]

    @classmethod
    def from_paths(
        cls,
        checkpoint_path: Path,
        metadata_path: Path,
        requested_device: str = "auto",
    ) -> "ArtifactPredictor":
        return cls(
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            device=select_device(requested_device),
        )

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != "efficientnet_b0":
            raise ValueError(f"Unsupported model: {checkpoint.get('model_name')}")
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid checkpoint without model_state_dict: {path}")
        return checkpoint

    def _validate_class_mapping(self, mapping: object) -> dict[str, int]:
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Checkpoint does not contain a valid class_to_idx mapping")
        normalized = {str(key): int(value) for key, value in mapping.items()}
        expected_indices = list(range(len(normalized)))
        if sorted(normalized.values()) != expected_indices:
            raise ValueError(f"Class indices must be contiguous: {normalized}")
        missing_metadata = sorted(set(normalized) - set(self.artifact_names))
        if missing_metadata:
            raise ValueError(f"Classes missing from metadata: {missing_metadata}")
        return normalized

    @staticmethod
    def _load_artifact_names(path: Path) -> dict[str, str]:
        if not path.is_file():
            raise FileNotFoundError(f"Metadata not found: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            names = {
                row["artifact_id"].strip(): row["artifact_name"].strip()
                for row in csv.DictReader(handle)
            }
        if not names:
            raise ValueError(f"No artifact metadata found in {path}")
        return names

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> list[RankedPrediction]:
        if not 1 <= top_k <= len(self.class_to_idx):
            raise ValueError(f"top_k must be between 1 and {len(self.class_to_idx)}")
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        confidences, indices = probabilities.topk(top_k)
        predictions: list[RankedPrediction] = []
        for confidence, class_index in zip(
            confidences.cpu().tolist(),
            indices.cpu().tolist(),
        ):
            artifact_id = self.idx_to_class[class_index]
            predictions.append(
                RankedPrediction(
                    artifact_id=artifact_id,
                    artifact_name=self.artifact_names[artifact_id],
                    confidence=float(confidence),
                )
            )
        return predictions

    def predict_path(self, image_path: Path, top_k: int = 3) -> list[RankedPrediction]:
        resolved_path = image_path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Image not found: {resolved_path}")
        with Image.open(resolved_path) as image:
            return self.predict_pil(image, top_k=top_k)
