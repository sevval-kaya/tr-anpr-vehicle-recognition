#!/usr/bin/env python
"""Collapse VMMRdb's fine-grained `<make>_<model>_<year>` ImageFolder
dataset into a coarser brand-only one (data/processed/vmmrdb_by_brand/
by default) — real-world photos, one folder per make, for a brand
classifier. Each brand's images are pooled across all of its raw
model/year classes and capped at --max-images-per-brand (deterministic
random sample) so a heavily-represented brand like Ford (870 raw
classes) doesn't swamp training relative to a thin one like Renault
(1 raw class) — see plaka.data.datasets.aggregate_images_by_brand.

Brands VMMRdb doesn't cover at all for this project's target market
(Renault/Opel/Citroen/Peugeot are present but tiny; Togg not at all —
it postdates the dataset) are exactly why this step is meant to be
followed by scripts/build_classifier_dataset.py --make-only --merge,
adding real Turkey-domain photos on top (see docs/decisions.md #25/#26).

    python scripts/aggregate_vmmrdb_by_brand.py

Defaults to configs/classification.yaml's target_makes_subset (the
already-curated Turkey-relevant make list) and its max cap.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml

from plaka.data.datasets import aggregate_images_by_brand
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VMMRDB_ROOT = REPO_ROOT / "data" / "external" / "vmmrdb"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "vmmrdb_by_brand"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "classification.yaml"
DEFAULT_MAX_IMAGES_PER_BRAND = 300


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def materialize(
    vmmrdb_root: Path,
    output_dir: Path,
    target_brands: list[str],
    max_images_per_brand: int,
    seed: int,
) -> dict[str, int]:
    pools = aggregate_images_by_brand(vmmrdb_root, target_brands, max_images_per_brand, seed)

    shutil.rmtree(output_dir, ignore_errors=True)
    counts: dict[str, int] = {}
    for brand, images in pools.items():
        brand_dir = output_dir / brand
        brand_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            # Namespace by source class dir to rule out any filename
            # collision across different raw model/year classes.
            dest_name = f"{image_path.parent.name}__{image_path.name}"
            shutil.copy2(image_path, brand_dir / dest_name)
        counts[brand] = len(images)

    missing = sorted(set(target_brands) - set(counts))
    logger.info(
        "materialized %d image(s) across %d brand(s) -> %s%s",
        sum(counts.values()),
        len(counts),
        output_dir,
        f" (not found in VMMRdb at all: {', '.join(missing)})" if missing else "",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmmrdb-root", type=Path, default=DEFAULT_VMMRDB_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--max-images-per-brand", type=int, default=DEFAULT_MAX_IMAGES_PER_BRAND
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = _load_config(args.config)
    target_brands = config["target_makes_subset"]["target_makes"]

    counts = materialize(
        args.vmmrdb_root,
        args.output_dir,
        target_brands,
        args.max_images_per_brand,
        args.seed,
    )
    for brand in sorted(counts):
        print(f"  {brand}: {counts[brand]}")


if __name__ == "__main__":
    main()
