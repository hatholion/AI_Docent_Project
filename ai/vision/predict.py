"""Predict an artifact from one image and return JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.vision.config import PROJECT_ROOT
from ai.vision.inference import ArtifactPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "metadata.csv")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = ArtifactPredictor.from_paths(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        requested_device=args.device,
    )
    predictions = predictor.predict_path(args.image, top_k=args.top_k)
    best = predictions[0]
    payload = {
        "image": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_stage": predictor.checkpoint_stage,
        "checkpoint_epoch": predictor.checkpoint_epoch,
        "device": predictor.device.type,
        "artifact_id": best.artifact_id,
        "artifact_name": best.artifact_name,
        "confidence": best.confidence,
        "top_k": [prediction.to_dict() for prediction in predictions],
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
