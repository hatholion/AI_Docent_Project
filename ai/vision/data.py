"""ImageFolder datasets and data loaders for artifact classification."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from ai.vision.config import VisionConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataBundle:
    datasets: dict[str, datasets.ImageFolder]
    loaders: dict[str, DataLoader]
    class_to_idx: dict[str, int]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(config: VisionConfig) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(
                degrees=12,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.03,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.15,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                config.resize_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return {
        "train": train_transform,
        "val": evaluation_transform,
        "test": evaluation_transform,
    }


def load_metadata_ids(config: VisionConfig) -> list[str]:
    with config.metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        artifact_ids = [row["artifact_id"].strip() for row in rows]
    if not artifact_ids:
        raise ValueError(f"No artifact IDs found in {config.metadata_path}")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError(f"Duplicate artifact IDs found in {config.metadata_path}")
    return artifact_ids


def validate_class_mapping(
    split: str,
    dataset: datasets.ImageFolder,
    expected_ids: set[str],
    reference_mapping: dict[str, int] | None,
) -> None:
    actual_ids = set(dataset.classes)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Class mismatch in {split}: missing={missing}, unexpected={unexpected}"
        )
    if reference_mapping is not None and dataset.class_to_idx != reference_mapping:
        raise ValueError(
            f"Class index mismatch in {split}: "
            f"expected={reference_mapping}, actual={dataset.class_to_idx}"
        )


def build_data_bundle(
    config: VisionConfig,
    device: torch.device,
) -> DataBundle:
    seed_everything(config.seed)
    split_transforms = build_transforms(config)
    expected_ids = set(load_metadata_ids(config))

    split_datasets: dict[str, datasets.ImageFolder] = {}
    reference_mapping: dict[str, int] | None = None
    for split in ("train", "val", "test"):
        split_path = config.data_root / split
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_path}")
        dataset = datasets.ImageFolder(split_path, transform=split_transforms[split])
        validate_class_mapping(split, dataset, expected_ids, reference_mapping)
        if reference_mapping is None:
            reference_mapping = dataset.class_to_idx
        split_datasets[split] = dataset

    if reference_mapping is None:
        raise RuntimeError("No dataset splits were loaded")

    generator = torch.Generator().manual_seed(config.seed)
    split_loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.num_workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in split_datasets.items()
    }
    return DataBundle(split_datasets, split_loaders, reference_mapping)
