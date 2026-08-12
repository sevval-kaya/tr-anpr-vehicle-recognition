"""Dataset layout helpers shared by training and inference.

Classification datasets are expected in ImageFolder convention (one
subdirectory per class, e.g. `renault_clio_mk4/*.jpg`), since that's what
VMMRdb/Stanford Cars/CompCars naturally convert into and what
`torchvision.datasets.ImageFolder` / timm training scripts expect directly.
"""

from __future__ import annotations

from pathlib import Path


def discover_class_names(dataset_root: str | Path) -> list[str]:
    """Scan an ImageFolder-style directory and return sorted class names.

    Sorted order matches the label indices `torchvision.datasets.ImageFolder`
    would assign, so this can be used to regenerate the class list a model
    was trained against.

    Raises:
        FileNotFoundError: if dataset_root doesn't exist.
        ValueError: if dataset_root contains no subdirectories.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not class_names:
        raise ValueError(f"no class subdirectories found under {root}")
    return class_names


def write_class_names(class_names: list[str], output_path: str | Path) -> None:
    """Write one class name per line, in order — the format VehicleClassifier reads."""
    Path(output_path).write_text("\n".join(class_names) + "\n", encoding="utf-8")
