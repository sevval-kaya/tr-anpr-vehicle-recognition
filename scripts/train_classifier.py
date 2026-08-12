#!/usr/bin/env python
"""Train the vehicle make/model classifier (timm backbone) on an
ImageFolder-convention dataset such as VMMRdb.

Full-scale training (VMMRdb: 9,170 classes, ~285K images) needs GPU compute
this environment doesn't have (torch here is CPU-only) — use --max-classes
and --max-images-per-class for a fast CPU smoke test that proves the
pipeline works end to end; drop them for a real run on a GPU machine.

    # smoke test: 5 random classes, 20 images each, 1 epoch
    python scripts/train_classifier.py data/external/vmmrdb \
        --max-classes 5 --max-images-per-class 20 --epochs 1

    # full run (GPU recommended)
    python scripts/train_classifier.py data/external/vmmrdb

Writes the best checkpoint + class list to models/vehicle_classifier/,
matching the paths VehicleClassifier reads by default (configs/classification.yaml).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from plaka.data.datasets import discover_class_names, write_class_names
from plaka.evaluation.metrics import top_k_accuracy
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "classification.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "vehicle_classifier"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
VAL_FRACTION = 0.1


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """ImageFolder-convention dataset restricted to a given class list and,
    optionally, a per-class image cap — the cap is what makes a fast CPU
    smoke test on a slice of a large dataset possible without touching the
    on-disk layout.
    """

    def __init__(
        self,
        root: Path,
        class_names: list[str],
        transform: Any,
        max_images_per_class: int | None = None,
    ) -> None:
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        for class_name in class_names:
            image_paths = sorted(
                p for p in (root / class_name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_images_per_class is not None:
                image_paths = image_paths[:max_images_per_class]
            self.samples.extend((path, self.class_to_idx[class_name]) for path in image_paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        transformed: torch.Tensor = self.transform(image)
        return transformed, label


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _split_train_val(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    return indices[n_val:], indices[:n_val]


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    class_names: list[str],
) -> tuple[float, float]:
    model.eval()
    k = min(5, len(class_names))
    ranked_predictions: list[list[str]] = []
    references: list[str] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            top_indices = torch.topk(logits, k=k, dim=1).indices.cpu().tolist()
            ranked_predictions.extend([class_names[i] for i in row] for row in top_indices)
            references.extend(class_names[label] for label in labels.tolist())

    return top_k_accuracy(ranked_predictions, references, k=1), top_k_accuracy(
        ranked_predictions, references, k=k
    )


def train(
    data_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    max_classes: int | None,
    max_images_per_class: int | None,
    epochs_override: int | None,
    seed: int,
) -> None:
    device = torch.device(config["vehicle_classifier"]["device"])
    architecture = config["vehicle_classifier"]["architecture"]
    pretrained = bool(config["training"]["pretrained_backbone"])
    epochs = epochs_override or int(config["training"]["epochs_pretrain"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])

    all_class_names = discover_class_names(data_root)
    if max_classes is not None:
        sample_size = min(max_classes, len(all_class_names))
        class_names = sorted(random.Random(seed).sample(all_class_names, sample_size))
        logger.info(
            "Restricting to %d/%d classes (smoke test)", len(class_names), len(all_class_names)
        )
    else:
        class_names = all_class_names
    logger.info("Training on %d classes", len(class_names))

    model = timm.create_model(architecture, pretrained=pretrained, num_classes=len(class_names))
    model.to(device)

    # timm.data doesn't ship a py.typed marker for these helpers.
    data_config = timm.data.resolve_data_config(  # type: ignore[attr-defined,no-untyped-call]
        {}, model=model
    )
    train_transform = timm.data.create_transform(**data_config, is_training=True)  # type: ignore[attr-defined]
    eval_transform = timm.data.create_transform(**data_config, is_training=False)  # type: ignore[attr-defined]

    train_dataset = ImageFolderDataset(
        data_root, class_names, train_transform, max_images_per_class
    )
    eval_dataset = ImageFolderDataset(data_root, class_names, eval_transform, max_images_per_class)
    if len(train_dataset) == 0:
        raise ValueError(f"no images found under {data_root} for the selected classes")

    train_indices, val_indices = _split_train_val(len(train_dataset), VAL_FRACTION, seed)
    if not val_indices:
        val_indices = train_indices[:1]
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        Subset(eval_dataset, val_indices), batch_size=batch_size, shuffle=False
    )
    logger.info("train=%d val=%d images", len(train_indices), len(val_indices))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_top1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_indices)

        val_top1, val_top5 = _evaluate(model, val_loader, device, class_names)
        logger.info(
            "epoch %d/%d | train_loss=%.4f | val_top1=%.4f | val_top%d=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_top1,
            min(5, len(class_names)),
            val_top5,
        )

        if val_top1 >= best_val_top1:
            best_val_top1 = val_top1
            torch.save(model.state_dict(), output_dir / "best.pt")
            write_class_names(class_names, output_dir / "classes.txt")
            logger.info("New best checkpoint saved (val_top1=%.4f)", val_top1)

    logger.info("Training complete. Best val_top1=%.4f", best_val_top1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="ImageFolder-formatted dataset root (e.g. data/external/vmmrdb)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-classes", type=int, default=None, help="Restrict to N random classes (smoke test)."
    )
    parser.add_argument(
        "--max-images-per-class", type=int, default=None, help="Cap images per class (smoke test)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override configs/classification.yaml epochs_pretrain.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config=config,
        max_classes=args.max_classes,
        max_images_per_class=args.max_images_per_class,
        epochs_override=args.epochs,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
