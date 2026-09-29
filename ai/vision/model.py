"""EfficientNet-B0 model construction for artifact classification."""

from __future__ import annotations

import os

from torch import nn

from ai.vision.config import PROJECT_ROOT


os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    efficientnet_b0,
    efficientnet_b1,
)

# torchvision's recommended (crop_size, resize_size) per architecture, from each
# weights enum's `.transforms()` metadata. Used so evaluate.py/finetune.py can
# reproduce the resolution a checkpoint was trained at without guessing.
RECOMMENDED_IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "efficientnet_b0": (224, 256),
    "efficientnet_b1": (240, 255),
}


def create_efficientnet_b0(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model


def create_efficientnet_b1(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = efficientnet_b1(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model


def create_model(
    architecture: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if architecture == "efficientnet_b0":
        return create_efficientnet_b0(
            num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
        )
    if architecture == "efficientnet_b1":
        return create_efficientnet_b1(
            num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def unfreeze_last_feature_blocks(model: nn.Module, block_count: int) -> list[int]:
    """Unfreeze the last N top-level EfficientNet feature blocks."""
    if block_count < 1:
        raise ValueError("block_count must be at least 1")
    if not hasattr(model, "features"):
        raise TypeError("model does not expose EfficientNet features")

    feature_blocks = list(model.features.children())
    if block_count > len(feature_blocks):
        raise ValueError(
            f"block_count={block_count} exceeds available blocks={len(feature_blocks)}"
        )

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    unfrozen_indices = list(range(len(feature_blocks) - block_count, len(feature_blocks)))
    for index in unfrozen_indices:
        for parameter in feature_blocks[index].parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return unfrozen_indices
