"""Smoke-test ImageFolder loading and a six-class EfficientNet forward pass."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.vision.config import VisionConfig  # noqa: E402
from ai.vision.data import build_data_bundle  # noqa: E402
from ai.vision.model import create_efficientnet_b0  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip downloading ImageNet weights during the smoke test.",
    )
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")

    device = select_device()
    config = replace(VisionConfig(), batch_size=args.batch_size)
    bundle = build_data_bundle(config, device)

    print(f"selected_device={device.type}")
    print(f"class_to_idx={bundle.class_to_idx}")
    for split, dataset in bundle.datasets.items():
        print(f"{split}_images={len(dataset)}")

    images, labels = next(iter(bundle.loaders["train"]))
    print(f"batch_images_shape={tuple(images.shape)}")
    print(f"batch_labels_shape={tuple(labels.shape)}")

    model = create_efficientnet_b0(
        len(bundle.class_to_idx),
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    ).to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device, non_blocking=device.type == "cuda"))

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    print(f"logits_shape={tuple(logits.shape)}")
    print(f"trainable_parameters={trainable_parameters}")
    print(f"frozen_parameters={frozen_parameters}")
    print("vision_pipeline_smoke_test=ok")


if __name__ == "__main__":
    main()
