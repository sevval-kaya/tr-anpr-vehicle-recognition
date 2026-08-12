#!/usr/bin/env python
"""Download baseline open-source datasets into data/external/.

Each dataset has a different access mechanism (Kaggle API, git clone,
manual license request), so this exposes one subcommand per dataset rather
than a single generic "download everything" that would silently fail on
whichever one needs manual steps.

Nothing here runs automatically — it's invoked explicitly, e.g.:

    python scripts/download_datasets.py vmmrdb
    python scripts/download_datasets.py turkish-plates
    python scripts/download_datasets.py stanford-cars

CompCars is not automated: its license requires filling out a request form
per-institution (see docs/decisions.md), so it's documented but not fetched
here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from plaka.utils.logging import get_logger

logger = get_logger(__name__)

EXTERNAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external"

VMMRDB_REPO_URL = "https://github.com/lgov/VMMRdb.git"
TURKISH_PLATE_KAGGLE_SLUG = "smaildurcan/turkish-license-plate-dataset"


def download_vmmrdb(destination: Path) -> None:
    """Clone VMMRdb (9,170 classes, ~291,752 images, 1950-2016).

    This is a large clone (multi-GB with LFS-tracked images); expect it to
    take a while on a slow connection.
    """
    if destination.exists():
        logger.info("VMMRdb already present at %s, skipping clone", destination)
        return
    logger.info("Cloning VMMRdb into %s", destination)
    subprocess.run(["git", "clone", VMMRDB_REPO_URL, str(destination)], check=True)


def download_turkish_plates(destination: Path) -> None:
    """Download the Kaggle Turkish License Plate Dataset via the Kaggle CLI.

    Requires `kaggle` to be installed and `~/.kaggle/kaggle.json` (or
    KAGGLE_USERNAME/KAGGLE_KEY env vars) configured with valid API
    credentials.
    """
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s via Kaggle CLI into %s", TURKISH_PLATE_KAGGLE_SLUG, destination)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            TURKISH_PLATE_KAGGLE_SLUG,
            "-p",
            str(destination),
            "--unzip",
        ],
        check=True,
    )


def download_stanford_cars(destination: Path) -> None:
    """Download Stanford Cars (196 classes) via torchvision's built-in dataset."""
    try:
        from torchvision.datasets import StanfordCars
    except ImportError:
        logger.error(
            "torchvision is required for Stanford Cars; install with "
            "`pip install -e '.[classification]'`"
        )
        raise
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Stanford Cars into %s", destination)
    StanfordCars(root=str(destination), download=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=["vmmrdb", "turkish-plates", "stanford-cars"],
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Override the default data/external/<dataset> destination.",
    )
    args = parser.parse_args(argv)

    default_destinations = {
        "vmmrdb": EXTERNAL_DATA_DIR / "vmmrdb",
        "turkish-plates": EXTERNAL_DATA_DIR / "turkish_plates",
        "stanford-cars": EXTERNAL_DATA_DIR / "stanford_cars",
    }
    destination = args.destination or default_destinations[args.dataset]

    downloaders = {
        "vmmrdb": download_vmmrdb,
        "turkish-plates": download_turkish_plates,
        "stanford-cars": download_stanford_cars,
    }
    downloaders[args.dataset](destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
