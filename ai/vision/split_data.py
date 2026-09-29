"""Build data/processed from data/raw. data/raw is only read, never modified.

synthetic images -> train, real images -> val/test (split per class).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from ai.vision.config import PROJECT_ROOT, VisionConfig
from ai.vision.data import load_metadata_ids

# Same extensions that torchvision ImageFolder accepts.
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp",
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-root", type=Path, default=VisionConfig().data_root)
    parser.add_argument("--seed", type=int, default=VisionConfig().seed)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split plan only; copy nothing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previously copied files in train/val/test (folders are kept) "
        "and copy again.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    skipped = [path.name for path in files if path.suffix.lower() not in IMAGE_SUFFIXES]
    if skipped:
        print(f"[warn] {directory}: skipped non-image files {skipped}")
    return images


def class_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir()}


def find_previous_outputs(output_root: Path) -> list[Path]:
    found: list[Path] = []
    for split in SPLITS:
        split_dir = output_root / split
        if split_dir.is_dir():
            found.extend(path for path in split_dir.rglob("*") if path.is_file())
    manifest = output_root / "split_manifest.csv"
    if manifest.is_file():
        found.append(manifest)
    return found


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if (
        output_root == raw_root
        or raw_root in output_root.parents
        or output_root in raw_root.parents
    ):
        raise ValueError("output-root must be separate from raw-root")

    real_dirs = class_dirs(raw_root / "real")
    synthetic_dirs = class_dirs(raw_root / "synthetic")
    metadata_ids = set(load_metadata_ids(VisionConfig()))
    if set(real_dirs) != set(synthetic_dirs) or set(real_dirs) != metadata_ids:
        raise ValueError(
            "Class folders differ: "
            f"real={sorted(real_dirs)}, synthetic={sorted(synthetic_dirs)}, "
            f"metadata={sorted(metadata_ids)}"
        )

    # 1) Build the plan first (no file is touched yet).
    plan: list[tuple[str, str, Path]] = []  # (split, artifact_id, source)
    for artifact_id in sorted(real_dirs):
        real_images = list_images(real_dirs[artifact_id])
        synthetic_images = list_images(synthetic_dirs[artifact_id])
        if len(real_images) < 2:
            raise ValueError(f"{artifact_id}: need at least 2 real images for val/test")
        if not synthetic_images:
            raise ValueError(f"{artifact_id}: no synthetic images")

        shuffled = real_images[:]
        random.Random(f"{args.seed}:{artifact_id}").shuffle(shuffled)
        val_count = len(shuffled) // 2  # test always gets the larger half
        plan += [("val", artifact_id, path) for path in shuffled[:val_count]]
        plan += [("test", artifact_id, path) for path in shuffled[val_count:]]
        plan += [("train", artifact_id, path) for path in synthetic_images]

    counts = {a: {s: 0 for s in SPLITS} for a in sorted(real_dirs)}
    for split, artifact_id, _ in plan:
        counts[artifact_id][split] += 1
    print(f"{'artifact_id':<12}{'train':>7}{'val':>6}{'test':>6}")
    for artifact_id, row in counts.items():
        print(f"{artifact_id:<12}{row['train']:>7}{row['val']:>6}{row['test']:>6}")
    print(
        f"{'total':<12}"
        f"{sum(r['train'] for r in counts.values()):>7}"
        f"{sum(r['val'] for r in counts.values()):>6}"
        f"{sum(r['test'] for r in counts.values()):>6}"
    )
    if args.dry_run:
        print("dry-run: nothing was copied")
        return

    # 2) Previously copied files are removed only with --overwrite (folders stay).
    previous = find_previous_outputs(output_root)
    if previous:
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already has {len(previous)} files. "
                "Use --overwrite to replace them."
            )
        for path in previous:
            path.unlink()

    # 3) Copy only; sources are never moved or deleted.
    manifest_rows: list[dict[str, str]] = []
    for split, artifact_id, source in plan:
        destination_dir = output_root / split / artifact_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)
        manifest_rows.append(
            {
                "split": split,
                "artifact_id": artifact_id,
                "domain": "synthetic" if split == "train" else "real",
                "source_path": str(source),
            }
        )

    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "artifact_id", "domain", "source_path")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"copied={len(plan)} manifest={manifest_path}")


if __name__ == "__main__":
    main()