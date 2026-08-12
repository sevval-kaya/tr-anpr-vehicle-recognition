"""YOLO-format object detection dataset utilities: discovery, splitting, and
normalization into this project's standard `data/processed/<name>/` layout
(`{train,val,test}/{images,labels}/` + `data.yaml`).

Source directories can be in any YOLO layout that nests `images/` and a
sibling `labels/` somewhere under them — this covers both a flat
`images/`+`labels/` source and Roboflow's per-split
`train/images/`+`train/labels/`, `valid/...`, `test/...` export layout, so
multiple sources (Roboflow export, Kaggle export, our own collected data)
can be merged and re-split consistently regardless of how each one arrived.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def _long_path(path: Path) -> str:
    """Prefix an absolute path with \\\\?\\ on Windows to opt into the Win32
    extended-length path API (no ~260 char MAX_PATH limit). Source datasets
    (e.g. Roboflow exports) can contain filenames derived from long
    social-media captions/hashtags that, combined with a deeply nested
    project directory, exceed the legacy limit during a plain copy.
    """
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


@dataclass(frozen=True, slots=True)
class YoloExample:
    """One image and its YOLO-format label file (may not exist on disk:
    a missing label file is a valid "no objects in this image" case)."""

    image_path: Path
    label_path: Path


def find_yolo_examples(source_dir: Path) -> list[YoloExample]:
    """Find every image/label pair under any `images/` directory in `source_dir`.

    Raises:
        FileNotFoundError: if source_dir doesn't exist.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {source_dir}")

    examples: list[YoloExample] = []
    for images_dir in sorted(source_dir.rglob("images")):
        labels_dir = images_dir.parent / "labels"
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            examples.append(
                YoloExample(
                    image_path=image_path,
                    label_path=labels_dir / f"{image_path.stem}.txt",
                )
            )
    return examples


def split_examples(
    examples: list[YoloExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[YoloExample]]:
    """Shuffle and split into train/val/test; test gets whatever ratio remains.

    Raises:
        ValueError: if examples is empty, or the ratios don't leave a
            positive share for the test split.
    """
    if not examples:
        raise ValueError("cannot split an empty example list")
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or train_ratio + val_ratio >= 1:
        raise ValueError(
            f"train_ratio ({train_ratio}) and val_ratio ({val_ratio}) must each be in "
            "(0, 1) and sum to less than 1, leaving a positive share for test"
        )

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def materialize_split(
    split: dict[str, list[YoloExample]],
    output_dir: Path,
    class_names: list[str],
) -> Path:
    """Copy each split into `output_dir/<split>/{images,labels}/` and write data.yaml.

    Images with no label file get an empty `.txt` written (explicit
    "no objects" rather than a silently missing file).

    Returns:
        Path to the written data.yaml (an Ultralytics-compatible training config).
    """
    for split_name, split_examples in split.items():
        images_out = output_dir / split_name / "images"
        labels_out = output_dir / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for example in split_examples:
            shutil.copy2(
                _long_path(example.image_path),
                _long_path(images_out / example.image_path.name),
            )
            label_dest = labels_out / f"{example.image_path.stem}.txt"
            if example.label_path.exists():
                shutil.copy2(_long_path(example.label_path), _long_path(label_dest))
            else:
                open(_long_path(label_dest), "wb").close()

    data_yaml_path = output_dir / "data.yaml"
    _write_data_yaml(data_yaml_path, output_dir, class_names)
    return data_yaml_path


def _write_data_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    content = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
