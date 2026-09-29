"""Build data/processed from data/raw. data/raw is only read, never modified.

synthetic images -> train, real images -> val/test (split per class).

Real images per class are scarce (as few as 3), so a single fixed val/test
split gives unreliable metrics. Instead this script builds `n_splits` folds:
for fold i, one rotating subset of each class's real images becomes `test`
and the remaining real images become `val`. `train` (synthetic) is identical
across folds. Run finetune.py/evaluate.py once per fold (--data-root pointing
at data/processed/fold_<i>) and average the resulting metrics.
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
FOLD_GLOB = "fold_*"
FINAL_DIR_NAME = "final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-root", type=Path, default=VisionConfig().data_root)
    parser.add_argument("--seed", type=int, default=VisionConfig().seed)
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of rotating val/test folds built from the real images. "
        "Must not exceed the smallest per-class real image count.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Build data/processed/final instead of fold_* folders: all real + all "
        "synthetic images go to train, except --sanity-count real images per class "
        "held out as a val/test sanity check. Use this for the deployed model, once "
        "the fold_* runs have validated the approach.",
    )
    parser.add_argument(
        "--sanity-count",
        type=int,
        default=1,
        help="Real images per class withheld from train in --final mode, used as a "
        "small sanity-check val/test set (not a rigorous evaluation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split plan only; copy nothing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previously copied fold_* (or final/) folders and copy again.",
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


def fold_dir(output_root: Path, fold_index: int) -> Path:
    return output_root / f"fold_{fold_index}"


def find_previous_outputs(output_root: Path) -> list[Path]:
    found: list[Path] = []
    if not output_root.is_dir():
        return found
    for path in output_root.glob(FOLD_GLOB):
        if path.is_dir():
            found.extend(item for item in path.rglob("*") if item.is_file())
    return found


def assign_folds(count: int, n_splits: int, rng: random.Random) -> list[int]:
    """Return a per-image fold id (0..n_splits-1), balanced within 1 image."""
    order = list(range(count))
    rng.shuffle(order)
    fold_of_position = [position % n_splits for position in range(count)]
    fold_ids = [0] * count
    for position, image_index in enumerate(order):
        fold_ids[image_index] = fold_of_position[position]
    return fold_ids


def build_fold_plan(
    real_dirs: dict[str, Path],
    synthetic_dirs: dict[str, Path],
    n_splits: int,
    seed: int,
) -> dict[int, list[tuple[str, str, Path]]]:
    plans: dict[int, list[tuple[str, str, Path]]] = {fold: [] for fold in range(n_splits)}
    for artifact_id in sorted(real_dirs):
        real_images = list_images(real_dirs[artifact_id])
        synthetic_images = list_images(synthetic_dirs[artifact_id])
        if len(real_images) < n_splits:
            raise ValueError(
                f"{artifact_id}: only {len(real_images)} real images, "
                f"need at least n_splits={n_splits}"
            )
        if not synthetic_images:
            raise ValueError(f"{artifact_id}: no synthetic images")

        rng = random.Random(f"{seed}:{artifact_id}")
        fold_ids = assign_folds(len(real_images), n_splits, rng)
        for fold in range(n_splits):
            test_images = [
                path for path, image_fold in zip(real_images, fold_ids) if image_fold == fold
            ]
            val_images = [
                path for path, image_fold in zip(real_images, fold_ids) if image_fold != fold
            ]
            plans[fold] += [("test", artifact_id, path) for path in test_images]
            plans[fold] += [("val", artifact_id, path) for path in val_images]
            plans[fold] += [("train", artifact_id, path) for path in synthetic_images]
    return plans


def build_final_plan(
    real_dirs: dict[str, Path],
    synthetic_dirs: dict[str, Path],
    sanity_count: int,
    seed: int,
) -> list[tuple[str, str, Path]]:
    plan: list[tuple[str, str, Path]] = []
    for artifact_id in sorted(real_dirs):
        real_images = list_images(real_dirs[artifact_id])
        synthetic_images = list_images(synthetic_dirs[artifact_id])
        if len(real_images) <= sanity_count:
            raise ValueError(
                f"{artifact_id}: only {len(real_images)} real images, "
                f"need more than sanity_count={sanity_count}"
            )
        if not synthetic_images:
            raise ValueError(f"{artifact_id}: no synthetic images")

        shuffled = real_images[:]
        random.Random(f"{seed}:{artifact_id}").shuffle(shuffled)
        sanity_images = shuffled[:sanity_count]
        train_real_images = shuffled[sanity_count:]

        # Sanity images serve as both val and test: this is not a held-out
        # evaluation set (that already happened via fold_*), just a smoke check
        # that training is behaving normally.
        plan += [("val", artifact_id, path) for path in sanity_images]
        plan += [("test", artifact_id, path) for path in sanity_images]
        plan += [("train", artifact_id, path) for path in train_real_images]
        plan += [("train", artifact_id, path) for path in synthetic_images]
    return plan


def print_fold_summary(
    fold: int, plan: list[tuple[str, str, Path]], artifact_ids: list[str]
) -> None:
    counts = {artifact_id: {split: 0 for split in SPLITS} for artifact_id in artifact_ids}
    for split, artifact_id, _ in plan:
        counts[artifact_id][split] += 1
    print(f"--- fold {fold} ---")
    print(f"{'artifact_id':<12}{'train':>7}{'val':>6}{'test':>6}")
    for artifact_id in artifact_ids:
        row = counts[artifact_id]
        print(f"{artifact_id:<12}{row['train']:>7}{row['val']:>6}{row['test']:>6}")
    print(
        f"{'total':<12}"
        f"{sum(row['train'] for row in counts.values()):>7}"
        f"{sum(row['val'] for row in counts.values()):>6}"
        f"{sum(row['test'] for row in counts.values()):>6}"
    )


def print_final_summary(plan: list[tuple[str, str, Path]], artifact_ids: list[str]) -> None:
    counts = {artifact_id: {split: 0 for split in SPLITS} for artifact_id in artifact_ids}
    for split, artifact_id, _ in plan:
        counts[artifact_id][split] += 1
    print("--- final ---")
    print(f"{'artifact_id':<12}{'train':>7}{'val':>6}{'test':>6}")
    for artifact_id in artifact_ids:
        row = counts[artifact_id]
        print(f"{artifact_id:<12}{row['train']:>7}{row['val']:>6}{row['test']:>6}")
    print(
        f"{'total':<12}"
        f"{sum(row['train'] for row in counts.values()):>7}"
        f"{sum(row['val'] for row in counts.values()):>6}"
        f"{sum(row['test'] for row in counts.values()):>6}"
    )


def find_final_outputs(output_root: Path) -> list[Path]:
    final_root = output_root / FINAL_DIR_NAME
    if not final_root.is_dir():
        return []
    return [path for path in final_root.rglob("*") if path.is_file()]


def write_split(destination_root: Path, plan: list[tuple[str, str, Path]]) -> Path:
    manifest_rows: list[dict[str, str]] = []
    for split, artifact_id, source in plan:
        destination_dir = destination_root / split / artifact_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)
        manifest_rows.append(
            {
                "split": split,
                "artifact_id": artifact_id,
                "domain": "real" if "real" in source.parts else "synthetic",
                "source_path": str(source),
            }
        )

    manifest_path = destination_root / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("split", "artifact_id", "domain", "source_path")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("n-splits must be at least 2")
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

    artifact_ids = sorted(real_dirs)

    if args.final:
        # 1) Build the plan first (no file is touched yet).
        plan = build_final_plan(real_dirs, synthetic_dirs, args.sanity_count, args.seed)
        print_final_summary(plan, artifact_ids)

        if args.dry_run:
            print("dry-run: nothing was copied")
            return

        # 2) A previously copied final/ folder is removed only with --overwrite.
        previous = find_final_outputs(output_root)
        if previous:
            if not args.overwrite:
                raise FileExistsError(
                    f"{output_root / FINAL_DIR_NAME} already has {len(previous)} files. "
                    "Use --overwrite to replace them."
                )
            shutil.rmtree(output_root / FINAL_DIR_NAME)

        # 3) Copy only; sources are never moved or deleted.
        manifest_path = write_split(output_root / FINAL_DIR_NAME, plan)
        print(f"final copied={len(plan)} manifest={manifest_path}")
        return

    # 1) Build the plan first (no file is touched yet).
    plans = build_fold_plan(real_dirs, synthetic_dirs, args.n_splits, args.seed)
    for fold in range(args.n_splits):
        print_fold_summary(fold, plans[fold], artifact_ids)

    if args.dry_run:
        print("dry-run: nothing was copied")
        return

    # 2) Previously copied fold_* folders are removed only with --overwrite.
    previous = find_previous_outputs(output_root)
    if previous:
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already has {len(previous)} files under fold_* folders. "
                "Use --overwrite to replace them."
            )
        for path in output_root.glob(FOLD_GLOB):
            if path.is_dir():
                shutil.rmtree(path)

    # 3) Copy only; sources are never moved or deleted.
    total_copied = 0
    for fold in range(args.n_splits):
        manifest_path = write_split(fold_dir(output_root, fold), plans[fold])
        total_copied += len(plans[fold])
        print(f"fold={fold} copied={len(plans[fold])} manifest={manifest_path}")
    print(f"total_copied={total_copied} folds={args.n_splits}")


if __name__ == "__main__":
    main()
