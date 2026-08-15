#!/usr/bin/env python
"""Train the vehicle make/model classifier (timm backbone) on an
ImageFolder-convention dataset such as VMMRdb.

Full-scale training (VMMRdb: 9,170 classes, ~285K images) needs GPU compute
this environment doesn't have (torch here is CPU-only). For a CPU-feasible
baseline, use --turkey-subset: it restricts training to a Turkiye-relevant
make subset (configs/classification.yaml: target_makes_subset), selected
round-robin across makes rather than dominated by whichever make VMMRdb
happens to have the most raw classes for (see docs/decisions.md #13),
combined with a lightweight backbone, a frozen pretrained body (only the
classifier head trains), and a reduced input size.

    # CPU-feasible baseline: ~200 Turkiye-relevant classes, frozen backbone
    python scripts/train_classifier.py data/external/vmmrdb --turkey-subset

    # smoke test: 5 random classes, 20 images each, 1 epoch
    python scripts/train_classifier.py data/external/vmmrdb \
        --max-classes 5 --max-images-per-class 20 --epochs 1

    # full run, all classes, unfrozen (GPU recommended)
    python scripts/train_classifier.py data/external/vmmrdb --no-freeze-backbone

Writes the best checkpoint + class list to models/vehicle_classifier/,
matching the paths VehicleClassifier reads by default (configs/classification.yaml).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, cast

import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from plaka.data.datasets import discover_class_names, select_target_classes, write_class_names
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


def _freeze_backbone(model: torch.nn.Module) -> int:
    """Freeze every parameter except the classifier head (transfer-learning
    mode: only the head is trained, which is what makes a reasonably sized
    subset trainable on CPU in a practical amount of time).

    Returns the number of trainable parameters left, for logging.
    """
    # torch's nn.Module stub types attribute access as `Tensor | Module`
    # (its real __getattr__ can return either), so a timm-specific method
    # like get_classifier() needs an explicit Any escape hatch.
    head: torch.nn.Module = cast(Any, model).get_classifier()
    head_param_ids = {id(p) for p in head.parameters()}
    for param in model.parameters():
        param.requires_grad = id(param) in head_param_ids
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _unfreeze_last_block(model: torch.nn.Module) -> int:
    """Freeze everything except the classifier head, the final MBConv
    stage (`blocks[-1]`), and the head convolution that feeds it
    (`conv_head`/`bn2`) — a lighter fine-tune than unfreezing the whole
    backbone. On a small dataset, giving the last (most task-specific)
    stage room to adapt while keeping the early, more generic
    edge/texture layers fixed is a common middle ground between "frozen
    everywhere" (may underfit a domain-shifted target) and "fully
    unfrozen" (overfits fast with little data).

    Only implemented for timm's EfficientNet-family module layout
    (conv_stem/bn1/blocks/conv_head/bn2/classifier) — see
    docs/decisions.md #27.

    Returns the number of trainable parameters left, for logging.
    """
    m = cast(Any, model)
    trainable_modules = [m.blocks[-1], m.conv_head, m.bn2, m.get_classifier()]
    trainable_param_ids = {id(p) for module in trainable_modules for p in module.parameters()}
    for param in model.parameters():
        param.requires_grad = id(param) in trainable_param_ids
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_train_transform(data_config: dict[str, Any], image_size: int) -> Any:
    """Explicit, strong augmentation for small datasets: random-resized
    crop, color jitter, horizontal flip, and a slight rotation — timm's
    default `create_transform(is_training=True)` only does crop+flip,
    which isn't much of a regularizer when there are only a handful of
    images per class (see docs/decisions.md #27).
    """
    mean = data_config["mean"]
    std = data_config["std"]
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


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
    turkey_subset: bool,
    max_classes: int | None,
    max_images_per_class: int | None,
    epochs_override: int | None,
    image_size_override: int | None,
    freeze_backbone_override: bool | None,
    freeze_mode: str | None,
    strong_augmentation: bool,
    patience: int | None,
    num_workers: int,
    seed: int,
) -> None:
    device = torch.device(config["vehicle_classifier"]["device"])
    architecture = config["vehicle_classifier"]["architecture"]
    pretrained = bool(config["training"]["pretrained_backbone"])
    if freeze_mode is None:
        freeze_backbone = (
            freeze_backbone_override
            if freeze_backbone_override is not None
            else bool(config["training"].get("freeze_backbone", False))
        )
        freeze_mode = "full" if freeze_backbone else "none"
    epochs = epochs_override or int(config["training"]["epochs_pretrain"])
    batch_size = int(config["training"]["batch_size"])
    learning_rate = float(config["training"]["learning_rate"])
    image_size = image_size_override or int(config["training"]["image_size"])

    all_class_names = discover_class_names(data_root)
    if turkey_subset:
        subset_config = config["target_makes_subset"]
        target_count = max_classes or int(subset_config["max_classes"])
        class_names = select_target_classes(
            data_root, target_makes=subset_config["target_makes"], max_classes=target_count
        )
        logger.info(
            "Turkiye-relevant subset: %d/%d classes (makes: %s)",
            len(class_names),
            len(all_class_names),
            ", ".join(subset_config["target_makes"]),
        )
    elif max_classes is not None:
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

    if freeze_mode == "full":
        trainable_params = _freeze_backbone(model)
        logger.info(
            "Backbone frozen; %d trainable parameters (classifier head only)", trainable_params
        )
    elif freeze_mode == "partial":
        trainable_params = _unfreeze_last_block(model)
        logger.info(
            "Backbone partially frozen; %d trainable parameters "
            "(last block + conv_head + classifier)",
            trainable_params,
        )
    elif freeze_mode != "none":
        raise ValueError(f"freeze_mode must be one of full/partial/none, got {freeze_mode!r}")

    # timm.data doesn't ship a py.typed marker for these helpers.
    data_config = timm.data.resolve_data_config(  # type: ignore[attr-defined,no-untyped-call]
        {"input_size": (3, image_size, image_size)}, model=model
    )
    train_transform = (
        _build_train_transform(data_config, image_size)
        if strong_augmentation
        else timm.data.create_transform(**data_config, is_training=True)  # type: ignore[attr-defined]
    )
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
    # Image decode+resize is CPU-bound; without worker processes the GPU
    # sits mostly idle waiting on data (observed ~10% utilization on a
    # single-process loader with this dataset size).
    loader_kwargs: dict[str, Any] = (
        {"num_workers": num_workers, "persistent_workers": True} if num_workers > 0 else {}
    )
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        Subset(eval_dataset, val_indices), batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    logger.info("train=%d val=%d images", len(train_indices), len(val_indices))

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_top1 = 0.0
    epochs_since_improvement = 0

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
            # >= (not >) so a tie still re-saves the latest weights within
            # a plateau — but that means a tie must NOT reset the early
            # stopping counter below, or a metric stuck at the same value
            # (easy with a val set this small — e.g. 6 images means only
            # 7 possible val_top1 values) would never trigger it.
            strictly_improved = val_top1 > best_val_top1
            best_val_top1 = val_top1
            torch.save(model.state_dict(), output_dir / "best.pt")
            write_class_names(class_names, output_dir / "classes.txt")
            logger.info("New best checkpoint saved (val_top1=%.4f)", val_top1)
            if strictly_improved:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
        else:
            epochs_since_improvement += 1

        if patience is not None and epochs_since_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d/%d (%d epoch(s) without val_top1 improvement)",
                epoch + 1,
                epochs,
                epochs_since_improvement,
            )
            break

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
    parser.add_argument(
        "--turkey-subset",
        action="store_true",
        help="Restrict training to configs/classification.yaml's target_makes_subset "
        "(Turkiye-relevant makes, balanced round-robin selection).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override configs/classification.yaml image_size.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze all but the classifier head (default: configs/classification.yaml). "
        "Superseded by --freeze-mode if that's also given.",
    )
    parser.add_argument(
        "--freeze-mode",
        choices=["full", "partial", "none"],
        default=None,
        help="'full' = classifier head only (like --freeze-backbone); 'partial' = also "
        "unfreeze the backbone's last block + conv_head (see docs/decisions.md #27); "
        "'none' = fully unfrozen. Overrides --freeze-backbone when given.",
    )
    parser.add_argument(
        "--strong-augmentation",
        action="store_true",
        help="Use explicit random-crop + color-jitter + flip + rotation augmentation "
        "instead of timm's default (crop+flip only) — recommended for small datasets.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop early if val_top1 doesn't improve for this many consecutive epochs "
        "(default: no early stopping, run the full --epochs budget).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="DataLoader worker processes for image decode/resize (0 = main-process only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config=config,
        turkey_subset=args.turkey_subset,
        max_classes=args.max_classes,
        max_images_per_class=args.max_images_per_class,
        epochs_override=args.epochs,
        image_size_override=args.image_size,
        freeze_backbone_override=args.freeze_backbone,
        freeze_mode=args.freeze_mode,
        strong_augmentation=args.strong_augmentation,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
