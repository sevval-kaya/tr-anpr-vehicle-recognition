#!/usr/bin/env python
"""Merge, deduplicate, and split the hand-labeled make/model CSVs under
data/external/user_plates/labels_manual/ into a clean train/held-out-test
pair, then report class balance.

    python scripts/prepare_manual_vehicle_labels.py

Writes train.csv / test_holdout.csv / SPLIT_MANIFEST.csv to --output-dir
(default data/processed/vehicle_labels_manual_split/) — the manifest lists
every usable image_file with its brand and which split it landed in, for
transparency (see docs/decisions.md #27).

Does not touch data/external/user_plates/images/ itself; run
scripts/build_classifier_dataset.py against the written train.csv to
materialize the actual ImageFolder training directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from plaka.data.manual_labels import load_label_csvs, merge_and_dedupe_labels, stratified_holdout_split
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_DIR = REPO_ROOT / "data" / "external" / "user_plates" / "labels_manual"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "vehicle_labels_manual_split"


def _write_label_csv(path: Path, labels: dict[str, tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_file", "make", "model", "visibility"])
        for image_file in sorted(labels, key=lambda f: int(Path(f).stem)):
            make, model = labels[image_file]
            writer.writerow([image_file, make, model, "clear"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_label_csvs(args.csv_dir)
    merged = merge_and_dedupe_labels(rows)

    if merged.conflicts:
        logger.warning("%d label conflict(s) found — excluded from the dataset:", len(merged.conflicts))
        for conflict in merged.conflicts:
            entries = ", ".join(f"{source}={make!r}" for source, make in conflict.entries)
            logger.warning("  %s: %s", conflict.image_file, entries)

    logger.info(
        "%d usable image(s), %d excluded (not_visible/blank make), %d conflict(s)",
        len(merged.usable),
        len(merged.excluded_not_visible),
        len(merged.conflicts),
    )

    train, test, single_image_brands = stratified_holdout_split(merged.usable, seed=args.seed)

    by_brand: dict[str, int] = {}
    for _f, (make, _model) in merged.usable.items():
        by_brand[make] = by_brand.get(make, 0) + 1

    print(f"\n{len(by_brand)} distinct brand(s):")
    for brand in sorted(by_brand, key=lambda b: -by_brand[b]):
        held_out = "no held-out test (1 image)" if brand in single_image_brands else "1 held out"
        print(f"  {brand:15s} n={by_brand[brand]:3d}  ({held_out})")

    if single_image_brands:
        print(
            f"\nBrands with only 1 image (train-only, no held-out test): "
            f"{', '.join(single_image_brands)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.csv"
    test_path = args.output_dir / "test_holdout.csv"
    _write_label_csv(train_path, train)
    _write_label_csv(test_path, test)

    manifest_path = args.output_dir / "SPLIT_MANIFEST.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_file", "make", "model", "split"])
        for image_file in sorted(train, key=lambda f: int(Path(f).stem)):
            make, model = train[image_file]
            writer.writerow([image_file, make, model, "train"])
        for image_file in sorted(test, key=lambda f: int(Path(f).stem)):
            make, model = test[image_file]
            writer.writerow([image_file, make, model, "test"])

    logger.info(
        "train=%d test=%d -> %s / %s (manifest: %s, seed=%d)",
        len(train),
        len(test),
        train_path,
        test_path,
        manifest_path,
        args.seed,
    )


if __name__ == "__main__":
    main()
