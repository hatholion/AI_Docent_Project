"""Evaluate an EfficientNet checkpoint and save classification diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from ai.vision.config import PROJECT_ROOT, VisionConfig
from ai.vision.data import build_data_bundle
from ai.vision.model import create_efficientnet_b0
from ai.vision.train import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--run-name", default="evaluation")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "vision" / "evaluations",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--no-copy-misclassified",
        action="store_true",
        help="Write the misclassification CSV without copying source images.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be at least 1")


def load_artifact_names(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["artifact_id"].strip(): row["artifact_name"].strip()
            for row in csv.DictReader(handle)
        }


def load_checkpoint_model(
    checkpoint_path: Path,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "efficientnet_b0":
        raise ValueError(f"Unsupported model: {checkpoint.get('model_name')}")
    if checkpoint.get("class_to_idx") != class_to_idx:
        raise ValueError(
            "Checkpoint class mapping does not match the dataset: "
            f"checkpoint={checkpoint.get('class_to_idx')}, dataset={class_to_idx}"
        )

    model = create_efficientnet_b0(
        len(class_to_idx),
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def save_confusion_matrix(
    matrix: object,
    artifact_ids: list[str],
    output_path: Path,
) -> None:
    figure_size = max(7, len(artifact_ids) * 1.2)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    display = ConfusionMatrixDisplay(matrix, display_labels=artifact_ids)
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Artifact classification confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)
    config = replace(
        VisionConfig(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    bundle = build_data_bundle(config, device)
    dataset = bundle.datasets[args.split]
    loader = bundle.loaders[args.split]
    checkpoint_path = args.checkpoint.resolve()
    model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        bundle.class_to_idx,
        device,
    )

    idx_to_class = {
        class_index: artifact_id
        for artifact_id, class_index in bundle.class_to_idx.items()
    }
    artifact_ids = [idx_to_class[index] for index in range(len(idx_to_class))]
    artifact_names = load_artifact_names(config.metadata_path)
    output_dir = args.output_root.resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    misclassified_dir = output_dir / "misclassified"
    if not args.no_copy_misclassified:
        misclassified_dir.mkdir(parents=True, exist_ok=True)

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    confidences: list[float] = []
    evaluated_paths: list[Path] = []
    sample_offset = 0

    print(f"selected_device={device.type}")
    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint_stage={checkpoint.get('training_stage', 'classifier_head')}")
    print(f"split={args.split}")
    with torch.inference_mode():
        progress = tqdm(loader, desc=f"evaluate {args.split}")
        for batch_index, (images, labels) in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch_size = labels.size(0)
            batch_samples = dataset.samples[sample_offset : sample_offset + batch_size]
            sample_offset += batch_size

            images = images.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            batch_confidences, predictions = probabilities.max(dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.cpu().tolist())
            confidences.extend(batch_confidences.cpu().tolist())
            evaluated_paths.extend(Path(path) for path, _ in batch_samples)

    if not true_labels:
        raise RuntimeError("No samples were evaluated")

    label_indices = list(range(len(artifact_ids)))
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=artifact_ids,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_indices)

    per_class_path = output_dir / "per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "artifact_id",
                "artifact_name",
                "precision",
                "recall",
                "f1_score",
                "support",
            ),
        )
        writer.writeheader()
        for artifact_id in artifact_ids:
            metrics = report[artifact_id]
            writer.writerow(
                {
                    "artifact_id": artifact_id,
                    "artifact_name": artifact_names.get(artifact_id, ""),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1-score"],
                    "support": int(metrics["support"]),
                }
            )

    misclassified_rows: list[dict[str, object]] = []
    for index, (path, true_index, predicted_index, confidence) in enumerate(
        zip(evaluated_paths, true_labels, predicted_labels, confidences),
        start=1,
    ):
        if true_index == predicted_index:
            continue
        true_id = idx_to_class[true_index]
        predicted_id = idx_to_class[predicted_index]
        copied_path = ""
        if not args.no_copy_misclassified:
            destination = misclassified_dir / (
                f"{index:04d}__true-{true_id}__pred-{predicted_id}__{path.name}"
            )
            shutil.copy2(path, destination)
            copied_path = str(destination)
        misclassified_rows.append(
            {
                "source_path": str(path),
                "copied_path": copied_path,
                "true_id": true_id,
                "true_name": artifact_names.get(true_id, ""),
                "predicted_id": predicted_id,
                "predicted_name": artifact_names.get(predicted_id, ""),
                "confidence": confidence,
            }
        )

    misclassified_path = output_dir / "misclassified.csv"
    with misclassified_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_path",
                "copied_path",
                "true_id",
                "true_name",
                "predicted_id",
                "predicted_name",
                "confidence",
            ),
        )
        writer.writeheader()
        writer.writerows(misclassified_rows)

    confusion_matrix_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(matrix, artifact_ids, confusion_matrix_path)
    summary_path = output_dir / "summary.json"
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("training_stage", "classifier_head"),
        "split": args.split,
        "samples": len(true_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "misclassified": len(misclassified_rows),
        "class_to_idx": bundle.class_to_idx,
        "synthetic_only": True,
        "max_batches": args.max_batches,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"samples={len(true_labels)}")
    print(f"accuracy={accuracy:.6f}")
    print(f"macro_f1={macro_f1:.6f}")
    print(f"misclassified={len(misclassified_rows)}")
    print(f"summary={summary_path}")
    print(f"per_class_metrics={per_class_path}")
    print(f"confusion_matrix={confusion_matrix_path}")
    print(f"misclassified_csv={misclassified_path}")


if __name__ == "__main__":
    main()
