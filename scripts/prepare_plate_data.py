#!/usr/bin/env python
"""Normalize one or more YOLO-format plate-detection sources into a single
train/val/test dataset under data/processed/<name>/.

Merges sources so a future Kaggle export or our own collected data can be
added to the same split alongside the Roboflow export, re-splitting all of
it together rather than keeping per-source splits inconsistent with each
other.

    python scripts/prepare_plate_data.py data/external/roboflow_plates
    python scripts/prepare_plate_data.py data/external/roboflow_plates data/external/kaggle_plates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from plaka.data.yolo_dataset import find_yolo_examples, materialize_split, split_examples
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "plates"
DEFAULT_CLASS_NAMES = ["license_plate"]


def _read_class_names(source_dirs: list[Path]) -> list[str]:
    """Read class names from the first source's data.yaml, if any; otherwise default."""
    for source_dir in source_dirs:
        data_yaml_path = source_dir / "data.yaml"
        if not data_yaml_path.exists():
            continue
        content = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        names = content.get("names") if isinstance(content, dict) else None
        if names:
            return list(names)
    logger.warning(
        "No data.yaml with class names found under any source; defaulting to %s",
        DEFAULT_CLASS_NAMES,
    )
    return list(DEFAULT_CLASS_NAMES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dirs",
        type=Path,
        nargs="+",
        help="One or more YOLO-format source directories (flat or per-split layout).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    examples = []
    for source_dir in args.source_dirs:
        found = find_yolo_examples(source_dir)
        logger.info("Found %d examples in %s", len(found), source_dir)
        examples.extend(found)

    if not examples:
        logger.error("No YOLO-format examples found in any source directory")
        return 1

    class_names = _read_class_names(args.source_dirs)
    split = split_examples(
        examples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    for split_name, split_examples_ in split.items():
        logger.info("%s: %d examples", split_name, len(split_examples_))

    data_yaml_path = materialize_split(split, args.output_dir, class_names)
    logger.info("Wrote %s", data_yaml_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
