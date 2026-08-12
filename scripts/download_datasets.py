#!/usr/bin/env python
"""Download baseline open-source datasets into data/external/.

Each dataset has a different access mechanism (direct zip download, Kaggle
API, manual license request), so this exposes one subcommand per dataset
rather than a single generic "download everything" that would silently fail
on whichever one needs manual steps.

Nothing here runs automatically — it's invoked explicitly, e.g.:

    python scripts/download_datasets.py vmmrdb
    python scripts/download_datasets.py turkish-plates             # Kaggle, needs kaggle.json
    python scripts/download_datasets.py turkish-plates-roboflow \
        --version 3                                                # needs ROBOFLOW_API_KEY
    python scripts/download_datasets.py stanford-cars

CompCars is not automated: its license requires filling out a request form
per-institution (see docs/decisions.md), so it's documented but not fetched
here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from plaka.utils.logging import get_logger

logger = get_logger(__name__)

EXTERNAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external"

# The VMMRdb git repo (github.com/faezetta/VMMRdb — NOT github.com/lgov/VMMRdb,
# which the source project brief cites but doesn't exist) holds only code and
# metadata; the ~291,752 images are hosted as a single ~11.5GB Dropbox zip.
VMMRDB_ZIP_URL = "https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=1"
TURKISH_PLATE_KAGGLE_SLUG = "smaildurcan/turkish-license-plate-dataset"
ROBOFLOW_WORKSPACE = "toggai"
ROBOFLOW_PROJECT = "turkish-license-plate"


def _download_with_progress(url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "plaka-dataset-downloader"})
    with urllib.request.urlopen(request) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        with (
            output_path.open("wb") as out_file,
            tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as progress,
        ):
            while chunk := response.read(1024 * 1024):
                out_file.write(chunk)
                progress.update(len(chunk))


def download_vmmrdb(destination: Path) -> None:
    """Download and extract VMMRdb (9,170 classes, ~291,752 images, 1950-2016).

    Downloads the ~11.5GB source zip with a progress bar, extracts it, then
    deletes the zip to avoid keeping two copies of the data on disk.
    """
    if destination.exists() and any(destination.iterdir()):
        logger.info("VMMRdb already present at %s, skipping download", destination)
        return
    destination.mkdir(parents=True, exist_ok=True)

    zip_path = destination.parent / "vmmrdb_download.zip"
    logger.info("Downloading VMMRdb (~11.5GB) to %s", zip_path)
    _download_with_progress(VMMRDB_ZIP_URL, zip_path)

    logger.info("Extracting %s into %s", zip_path, destination)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    zip_path.unlink()
    logger.info("VMMRdb ready at %s", destination)


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


def download_turkish_plates_roboflow(destination: Path, version: int) -> None:
    """Download the Roboflow "Turkish License Plate" dataset (YOLO-format, CC BY 4.0).

    Single class (license_plate), already bounding-box annotated — this is
    the primary source for plate *detector* training data (see
    docs/decisions.md). Requires a Roboflow API key: despite the permissive
    CC BY 4.0 license, Roboflow Universe has no anonymous bulk-download path
    — every export, public or not, goes through an API key tied to a
    (free) account. Get one at https://app.roboflow.com/settings/api and
    pass it via the ROBOFLOW_API_KEY environment variable.

    `version` must be looked up on the project page (the version selector
    dropdown) since Roboflow doesn't document a stable "latest" alias.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Get a free API key at "
            "https://app.roboflow.com/settings/api, then set it with "
            "`export ROBOFLOW_API_KEY=...` (or the Windows equivalent) "
            "before running this."
        )
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow is required; install with `pip install -e '.[data]'`")
        raise

    destination.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading Roboflow %s/%s v%d into %s",
        ROBOFLOW_WORKSPACE,
        ROBOFLOW_PROJECT,
        version,
        destination,
    )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(version).download("yolov11", location=str(destination))


def download_stanford_cars(destination: Path) -> None:
    """Download Stanford Cars (196 classes) via torchvision's built-in dataset.

    Known broken as of this writing: the original Stanford host is offline
    and torchvision's StanfordCars.download() raises ValueError
    unconditionally (see docs/decisions.md). Left in place in case
    torchvision points it at a working mirror in a future release; until
    then, use VMMRdb or set up a Kaggle mirror instead.
    """
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
        choices=["vmmrdb", "turkish-plates", "turkish-plates-roboflow", "stanford-cars"],
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Override the default data/external/<dataset> destination.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Roboflow dataset version number (turkish-plates-roboflow only; "
        "check the version dropdown on the project page).",
    )
    args = parser.parse_args(argv)

    default_destinations = {
        "vmmrdb": EXTERNAL_DATA_DIR / "vmmrdb",
        "turkish-plates": EXTERNAL_DATA_DIR / "kaggle_plates",
        "turkish-plates-roboflow": EXTERNAL_DATA_DIR / "roboflow_plates",
        "stanford-cars": EXTERNAL_DATA_DIR / "stanford_cars",
    }
    destination = args.destination or default_destinations[args.dataset]

    if args.dataset == "turkish-plates-roboflow":
        if args.version is None:
            parser.error("--version is required for turkish-plates-roboflow")
        download_turkish_plates_roboflow(destination, args.version)
        return 0

    downloaders = {
        "vmmrdb": download_vmmrdb,
        "turkish-plates": download_turkish_plates,
        "stanford-cars": download_stanford_cars,
    }
    downloaders[args.dataset](destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
