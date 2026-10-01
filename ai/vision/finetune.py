"""Fine-tune the last EfficientNet feature blocks from a classifier checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ai.vision.config import PROJECT_ROOT, VisionConfig
from ai.vision.data import build_data_bundle
from ai.vision.model import create_efficientnet_b0, unfreeze_last_feature_blocks
from ai.vision.train import (
    create_class_weights,
    run_epoch,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-blocks", type=int, default=2)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--run-name", default="fine_tune")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "vision")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.feature_blocks < 1:
        raise ValueError("epochs, batch-size, and feature-blocks must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.backbone_learning_rate <= 0 or args.classifier_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if args.weight_decay < 0 or args.patience < 1:
        raise ValueError("weight-decay must be non-negative and patience at least 1")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be at least 1")


def load_classifier_checkpoint(
    path: Path,
    model: nn.Module,
    expected_mapping: dict[str, int],
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "efficientnet_b0":
        raise ValueError(f"Unsupported model checkpoint: {checkpoint.get('model_name')}")
    if checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError(
            "Checkpoint class mapping does not match the current dataset: "
            f"checkpoint={checkpoint.get('class_to_idx')}, dataset={expected_mapping}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: object,
    class_to_idx: dict[str, int],
    config: VisionConfig,
    args: argparse.Namespace,
    source_checkpoint: Path,
    unfrozen_indices: list[int],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_name": "efficientnet_b0",
            "training_stage": "fine_tune",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": asdict(val_metrics),
            "class_to_idx": class_to_idx,
            "image_size": config.image_size,
            "resize_size": config.resize_size,
            "imagenet_pretrained": True,
            "backbone_frozen": False,
            "unfrozen_feature_blocks": unfrozen_indices,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "backbone_learning_rate": args.backbone_learning_rate,
            "classifier_learning_rate": args.classifier_learning_rate,
        },
        path,
    )


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

    model = create_efficientnet_b0(
        len(bundle.class_to_idx),
        pretrained=False,
        freeze_backbone=True,
    )
    source_checkpoint = args.checkpoint.resolve()
    source = load_classifier_checkpoint(
        source_checkpoint,
        model,
        bundle.class_to_idx,
    )
    unfrozen_indices = unfreeze_last_feature_blocks(model, args.feature_blocks)
    model.to(device)

    backbone_parameters = [
        parameter for parameter in model.features.parameters() if parameter.requires_grad
    ]
    classifier_parameters = list(model.classifier.parameters())
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": classifier_parameters, "lr": args.classifier_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=create_class_weights(bundle, device))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    output_dir = args.output_root.resolve() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.json"
    print(f"selected_device={device.type}")
    print(f"source_checkpoint={source_checkpoint}")
    print(f"source_epoch={source.get('epoch')}")
    print(f"unfrozen_feature_blocks={unfrozen_indices}")
    print(f"trainable_backbone_parameters={sum(p.numel() for p in backbone_parameters)}")
    print(f"trainable_classifier_parameters={sum(p.numel() for p in classifier_parameters)}")
    print(f"output_dir={output_dir}")

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
            description=f"fine-tune {epoch}/{args.epochs} train",
        )
        val_metrics = run_epoch(
            model=model,
            loader=bundle.loaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
            description=f"fine-tune {epoch}/{args.epochs} val",
        )
        scheduler.step(val_metrics.loss)
        backbone_lr = optimizer.param_groups[0]["lr"]
        classifier_lr = optimizer.param_groups[1]["lr"]
        history.append(
            {
                "epoch": epoch,
                "backbone_learning_rate": backbone_lr,
                "classifier_learning_rate": classifier_lr,
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_accuracy={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_accuracy={val_metrics.accuracy:.4f} "
            f"backbone_lr={backbone_lr:.6g} "
            f"classifier_lr={classifier_lr:.6g}"
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                class_to_idx=bundle.class_to_idx,
                config=config,
                args=args,
                source_checkpoint=source_checkpoint,
                unfrozen_indices=unfrozen_indices,
            )
            print(f"best_checkpoint_saved={checkpoint_path}")
        else:
            epochs_without_improvement += 1

        history_path.write_text(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "device": device.type,
                    "source_checkpoint": str(source_checkpoint),
                    "class_to_idx": bundle.class_to_idx,
                    "unfrozen_feature_blocks": unfrozen_indices,
                    "arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "epochs": history,
                    "elapsed_seconds": time.perf_counter() - started_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if epochs_without_improvement >= args.patience:
            print(f"early_stopping_epoch={epoch}")
            break

    print(f"fine_tuning_complete_elapsed_seconds={time.perf_counter() - started_at:.2f}")
    print(f"best_model={checkpoint_path}")
    print(f"history={history_path}")


if __name__ == "__main__":
    main()
