"""Split artifact images into ImageFolder train/val/test directories.

Synthetic images (data/raw/synthetic/<artifact_id>/) are split into train/val/test.
Real photos (data/raw/real/<artifact_id>/), when present, are never put into train and
are split into val/test only, weighted mostly toward test (see --real-val-ratio).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--real-val-ratio",
        type=float,
        default=0.20,
        help="Fraction of real photos assigned to val; the rest go to test (default 0.20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files that already exist.",
    )
    return parser.parse_args()


def load_artifact_ids(metadata_path: Path) -> list[str]:
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        artifact_ids = [row["artifact_id"].strip() for row in rows]

    if not artifact_ids or any(not artifact_id for artifact_id in artifact_ids):
        raise ValueError(f"No valid artifact IDs found in {metadata_path}")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError(f"Duplicate artifact IDs found in {metadata_path}")
    return artifact_ids


def split_images(
    images: list[Path], artifact_id: str, seed: int, train_ratio: float, val_ratio: float
) -> dict[str, list[Path]]:
    shuffled = sorted(images)
    random.Random(f"{seed}:{artifact_id}").shuffle(shuffled)

    train_end = round(len(shuffled) * train_ratio)
    val_end = train_end + round(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def split_real_images(
    images: list[Path], artifact_id: str, seed: int, val_ratio: float
) -> dict[str, list[Path]]:
    """Split real photos into val/test only (never train)."""
    shuffled = sorted(images)
    random.Random(f"{seed}:real:{artifact_id}").shuffle(shuffled)

    val_end = round(len(shuffled) * val_ratio)
    return {
        "train": [],
        "val": shuffled[:val_end],
        "test": shuffled[val_end:],
    }


def merge_split_files(
    synthetic: dict[str, list[Path]], real: dict[str, list[Path]]
) -> dict[str, list[Path]]:
    return {split: synthetic[split] + real[split] for split in SPLITS}


def main() -> None:
    args = parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("train and validation ratios must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1")
    if not 0 <= args.real_val_ratio <= 1:
        raise ValueError("real-val-ratio must be between 0 and 1")

    project_root = Path(__file__).resolve().parents[1]
    synthetic_root = project_root / "data" / "raw" / "synthetic"
    real_root = project_root / "data" / "raw" / "real"
    processed_root = project_root / "data" / "processed"
    artifact_ids = load_artifact_ids(project_root / "data" / "metadata.csv")

    totals = {split: 0 for split in SPLITS}
    real_totals = {split: 0 for split in SPLITS}
    for artifact_id in artifact_ids:
        source_dir = synthetic_root / artifact_id
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing raw class directory: {source_dir}")

        images = [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            raise ValueError(f"No images found in {source_dir}")

        split_files = split_images(
            images, artifact_id, args.seed, args.train_ratio, args.val_ratio
        )

        real_dir = real_root / artifact_id
        real_split_files = {"train": [], "val": [], "test": []}
        if real_dir.is_dir():
            real_images = [
                path
                for path in real_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if real_images:
                real_split_files = split_real_images(
                    real_images, artifact_id, args.seed, args.real_val_ratio
                )
                for split in SPLITS:
                    real_totals[split] += len(real_split_files[split])

        split_files = merge_split_files(split_files, real_split_files)

        counts: list[str] = []
        for split, files in split_files.items():
            destination_dir = processed_root / split / artifact_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            expected_names = {path.name for path in files}
            if args.overwrite:
                for existing_path in destination_dir.iterdir():
                    if (
                        existing_path.is_file()
                        and existing_path.suffix.lower() in IMAGE_EXTENSIONS
                        and existing_path.name not in expected_names
                    ):
                        existing_path.unlink()
            for source_path in files:
                destination_path = destination_dir / source_path.name
                if destination_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"Destination exists: {destination_path}. "
                        "Use --overwrite to regenerate the split."
                    )
                shutil.copy2(source_path, destination_path)
            totals[split] += len(files)
            counts.append(f"{split}={len(files)}")
        print(f"{artifact_id}: " + ", ".join(counts))

    print("total: " + ", ".join(f"{key}={value}" for key, value in totals.items()))
    print(
        "total (real only): "
        + ", ".join(f"{key}={value}" for key, value in real_totals.items())
    )


if __name__ == "__main__":
    main()
