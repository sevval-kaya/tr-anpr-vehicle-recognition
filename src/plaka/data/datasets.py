"""Dataset layout helpers shared by training and inference.

Classification datasets are expected in ImageFolder convention (one
subdirectory per class, e.g. `renault_clio_mk4/*.jpg`), since that's what
VMMRdb/Stanford Cars/CompCars naturally convert into and what
`torchvision.datasets.ImageFolder` / timm training scripts expect directly.
"""

from __future__ import annotations

import random
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


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def count_images_per_class(dataset_root: str | Path, class_names: list[str]) -> dict[str, int]:
    """Count image files directly under each class subdirectory."""
    root = Path(dataset_root)
    return {
        name: sum(1 for p in (root / name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for name in class_names
    }


def select_target_classes(
    dataset_root: str | Path,
    target_makes: list[str],
    max_classes: int,
) -> list[str]:
    """Select up to `max_classes` classes belonging to `target_makes`, balanced
    round-robin across makes rather than dominated by whichever make happens
    to have the most raw classes (e.g. VMMRdb has 870 Ford classes but only 1
    Renault — a plain top-N-by-count selection would drop rare-but-relevant
    makes entirely). Within each make, classes are ranked by image count so
    the best-populated model/year combinations are picked first.

    `target_makes` entries must match the dataset's class-name prefix
    convention exactly (VMMRdb separates make from the rest with `_`,
    except "mercedes benz" which uses a literal space — both are matched).

    Raises:
        FileNotFoundError: if dataset_root doesn't exist (via discover_class_names).
        ValueError: if no class matches any target make.
    """
    root = Path(dataset_root)
    all_class_names = discover_class_names(root)

    make_to_classes: dict[str, list[str]] = {make: [] for make in target_makes}
    makes_by_prefix_length = sorted(target_makes, key=len, reverse=True)
    for class_name in all_class_names:
        for make in makes_by_prefix_length:
            if class_name.startswith(f"{make}_") or class_name.startswith(f"{make} "):
                make_to_classes[make].append(class_name)
                break

    for classes_for_make in make_to_classes.values():
        counts = count_images_per_class(root, classes_for_make)
        classes_for_make.sort(key=lambda c: counts[c], reverse=True)

    selected: list[str] = []
    round_index = 0
    while len(selected) < max_classes:
        added_this_round = False
        for make in target_makes:
            classes_for_make = make_to_classes[make]
            if round_index < len(classes_for_make):
                selected.append(classes_for_make[round_index])
                added_this_round = True
                if len(selected) >= max_classes:
                    break
        if not added_this_round:
            break
        round_index += 1

    if not selected:
        raise ValueError(f"no classes under {root} matched target makes {target_makes}")

    return sorted(selected)


def extract_brand(class_name: str) -> str:
    """Extract the make/brand token from a `<make>_<model>_<year>`
    VMMRdb-convention class name.

    Splitting on the *first* underscore only is deliberate: it's the one
    rule that handles every VMMRdb class correctly, including "mercedes
    benz" (space-separated, no underscore inside the make itself, e.g.
    "mercedes benz_190_1985" -> "mercedes benz") without needing a
    special case for it.
    """
    return class_name.split("_", 1)[0]


def aggregate_images_by_brand(
    dataset_root: str | Path,
    target_brands: list[str],
    max_images_per_brand: int,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Pool every image across all of a brand's raw `<make>_<model>_<year>`
    class folders into one list per brand (restricted to `target_brands`),
    then cap each brand's pool at `max_images_per_brand` via a deterministic
    random sample — for collapsing a fine-grained (model+year) ImageFolder
    dataset like VMMRdb into a coarser brand-only one, so a brand with a
    couple of oversized model classes (e.g. Ford has 870 raw classes) doesn't
    dominate over a brand with a single tiny one (e.g. Renault has 1).

    Brands with fewer images than the cap contribute everything they have —
    the cap only ever removes images, never pads a sparse brand out.

    Returns {brand: [image_path, ...]}, only for brands that had at least
    one matching image (a target brand entirely absent from the dataset —
    e.g. a make VMMRdb never collected — is simply omitted, not an error).

    Raises:
        FileNotFoundError: if dataset_root doesn't exist.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    target_set = set(target_brands)
    pools: dict[str, list[Path]] = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        brand = extract_brand(class_dir.name)
        if brand not in target_set:
            continue
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        pools.setdefault(brand, []).extend(images)

    rng = random.Random(seed)
    capped: dict[str, list[Path]] = {}
    for brand, images in pools.items():
        if len(images) > max_images_per_brand:
            capped[brand] = rng.sample(images, max_images_per_brand)
        else:
            capped[brand] = images
    return capped
