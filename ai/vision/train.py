"""Train the classifier head of an ImageNet-pretrained EfficientNet-B0."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai.vision.config import PROJECT_ROOT, VisionConfig
from ai.vision.data import DataBundle, build_data_bundle
from ai.vision.model import create_efficientnet_b0


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=VisionConfig().data_root)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--run-name", default="classifier_head")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "vision")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet weights. Intended only for offline smoke tests.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be at least 1")


def create_class_weights(bundle: DataBundle, device: torch.device) -> torch.Tensor:
    targets = torch.tensor(bundle.datasets["train"].targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=len(bundle.class_to_idx)).float()
    if torch.any(counts == 0):
        raise ValueError(f"Every class needs train images, got counts={counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(device)


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    description: str,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    if training:
        # BatchNorm statistics in frozen blocks must not drift.
        for feature_block in model.features.children():
            if not any(
                parameter.requires_grad for parameter in feature_block.parameters()
            ):
                feature_block.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc=description, leave=False)

    for batch_index, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            accuracy=f"{total_correct / total_samples:.4f}",
        )

    if total_samples == 0:
        raise RuntimeError(f"No samples were processed for {description}")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
    )


def save_loss_curve(history: list[dict[str, object]], output_path: Path) -> None:
    epochs = [record["epoch"] for record in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epochs, [record["train"]["loss"] for record in history], label="train")
    loss_axis.plot(epochs, [record["val"]["loss"] for record in history], label="val")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.set_title("Loss")
    loss_axis.legend()

    accuracy_axis.plot(
        epochs, [record["train"]["accuracy"] for record in history], label="train"
    )
    accuracy_axis.plot(epochs, [record["val"]["accuracy"] for record in history], label="val")
    accuracy_axis.set_xlabel("epoch")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    class_to_idx: dict[str, int],
    config: VisionConfig,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "format_version": 1,
        "model_name": "efficientnet_b0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_metrics": asdict(metrics),
        "class_to_idx": class_to_idx,
        "image_size": config.image_size,
        "resize_size": config.resize_size,
        "imagenet_pretrained": not args.no_pretrained,
        "backbone_frozen": True,
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)
    config = replace(
        VisionConfig(),
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    bundle = build_data_bundle(config, device)
    model = create_efficientnet_b0(
        len(bundle.class_to_idx),
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=create_class_weights(bundle, device))
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    output_dir = args.output_root.resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.json"

    print(f"selected_device={device.type}")
    print(f"class_to_idx={bundle.class_to_idx}")
    print(f"output_dir={output_dir}")
    print(f"class_weights={criterion.weight.detach().cpu().tolist()}")

    best_val_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=bundle.loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            description=f"epoch {epoch}/{args.epochs} train",
        )
        val_metrics = run_epoch(
            model=model,
            loader=bundle.loaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
            description=f"epoch {epoch}/{args.epochs} val",
        )
        scheduler.step(val_metrics.loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_accuracy={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_accuracy={val_metrics.accuracy:.4f} "
            f"lr={learning_rate:.6g}"
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                class_to_idx=bundle.class_to_idx,
                config=config,
                args=args,
            )
            print(f"best_checkpoint_saved={checkpoint_path}")
        else:
            epochs_without_improvement += 1

        history_payload = {
            "run_name": args.run_name,
            "device": device.type,
            "class_to_idx": bundle.class_to_idx,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "epochs": history,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
        history_path.write_text(
            json.dumps(history_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if epochs_without_improvement >= args.patience:
            print(f"early_stopping_epoch={epoch}")
            break

    loss_curve_path = output_dir / "loss_curve.png"
    save_loss_curve(history, loss_curve_path)

    print(f"training_complete_elapsed_seconds={time.perf_counter() - started_at:.2f}")
    print(f"best_model={checkpoint_path}")
    print(f"history={history_path}")
    print(f"loss_curve={loss_curve_path}")


if __name__ == "__main__":
    main()
