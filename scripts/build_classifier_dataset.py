#!/usr/bin/env python
"""Materialize an ImageFolder-convention classifier dataset out of one or
more make/model label CSVs, for images that already live flat in a source
directory (e.g. data/external/user_plates/images/ — the same 1,955 images
used for the plate detector).

CSV format (header required), one row per image:

    image_file,make,model,visibility
    1041.jpg,ford,mustang,clear
    1250.jpg,honda,,clear
    253.jpg,,,not_visible

- `image_file`: filename inside --source-dir (must exist there).
- `make`, `model`: free text, any case/spacing — normalized to the
  project's class-name convention (lowercase, spaces/hyphens -> "_").
  Leave both blank when no make/model can be confidently identified.
- `visibility`: "clear" (logo/badge or unmistakable styling), "partial"
  (make readable, model a best guess/uncertain), or "not_visible" (skip
  this row). A blank `make` is always skipped. In the default make+model
  mode a blank `model` is also skipped (nothing to build a class from);
  in --make-only mode a blank model is fine — the row still counts
  towards its make's class. An `image_id` column, if present, is ignored
  (kept only for cross-referencing against other label CSVs from the
  same pilot).

Two output modes:

- **Fresh build** (default): wipes --output-dir and rebuilds it from
  scratch with `<make>_<model>` (or, with --make-only, `<make>`) class
  names, e.g. data/processed/vehicle_labels/. For a from-scratch
  Turkey-only dataset — pass straight to scripts/train_classifier.py (no
  --turkey-subset needed; every discovered class is already
  Turkey-relevant by construction).

- **Merge** (--merge): adds images into an *existing* ImageFolder
  directory (e.g. a downloaded Kaggle brand dataset's train/ split)
  without touching anything already there. A class already present as a
  subdirectory is matched case-insensitively and reused with its
  existing casing (so labeling "toyota" lands in an existing "Toyota/",
  not a new "toyota/"); a genuinely new class is created Title-Cased
  (e.g. "renault" -> "Renault/"). Never deletes or overwrites existing
  files. Use --dry-run first to see the plan without touching disk —
  recommended before pointing --merge at real external data.

Deliberately NOT included: model year / generation. It usually isn't
reliably determinable from a single photo, and this dataset is already
small — splitting further by year would leave most classes with a
handful of images each. Add a generation suffix later only for models
with a visually unmistakable styling break and enough samples on both
sides to be worth it.

    # Fresh Turkey-only make+model dataset
    python scripts/build_classifier_dataset.py labels.csv

    # Merge Turkey make-only labels into a downloaded Kaggle brand dataset
    python scripts/build_classifier_dataset.py pilot1.csv pilot2.csv pilot3.csv \\
        --make-only --merge --output-dir data/external/car_brand_dataset/train \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from plaka.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "external" / "user_plates" / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "vehicle_labels"
SKIP_VISIBILITY = frozenset({"not_visible"})

_NORMALIZE_PATTERN = re.compile(r"[\s\-]+")
_INVALID_CHARS_PATTERN = re.compile(r"[^a-z0-9_]")


def normalize_class_component(text: str) -> str:
    """Lowercase, collapse whitespace/hyphens to a single underscore, and
    drop anything outside [a-z0-9_] (e.g. accented characters, punctuation).
    """
    text = text.strip().lower()
    text = _NORMALIZE_PATTERN.sub("_", text)
    return _INVALID_CHARS_PATTERN.sub("", text)


def build_class_name(make: str, model: str, make_only: bool = False) -> str | None:
    """Combine make+model into a `make_model` class name (or just `make`
    in make_only mode), or None if a required field is blank after
    normalization.
    """
    make_norm = normalize_class_component(make)
    if not make_norm:
        return None
    if make_only:
        return make_norm
    model_norm = normalize_class_component(model)
    if not model_norm:
        return None
    return f"{make_norm}_{model_norm}"


def resolve_class_directory_name(class_key: str, raw_make: str, output_dir: Path) -> str:
    """For --merge mode: reuse an existing subdirectory of output_dir if
    one matches `class_key` case-insensitively (preserving its existing
    casing); otherwise Title-Case the raw make text for a new directory
    (e.g. "renault" -> "Renault"), consistent with how brand-only Kaggle
    datasets typically name their class folders.
    """
    if output_dir.is_dir():
        for existing in output_dir.iterdir():
            if existing.is_dir() and normalize_class_component(existing.name) == class_key:
                return existing.name
    return raw_make.strip().title()


def _collect_labeled_rows(
    csv_paths: list[Path],
    source_dir: Path,
    make_only: bool,
) -> tuple[dict[str, list[tuple[str, str]]], int, int]:
    """Read every CSV and group (image_file, raw_make) pairs by normalized
    class key. Returns (by_class_key, skipped_unlabeled, skipped_missing_file).
    """
    by_class_key: dict[str, list[tuple[str, str]]] = {}
    skipped_unlabeled = 0
    skipped_missing_file = 0

    for csv_path in csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"label CSV not found: {csv_path}")
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        for row in rows:
            if row.get("visibility", "").strip().lower() in SKIP_VISIBILITY:
                skipped_unlabeled += 1
                continue
            class_key = build_class_name(row.get("make", ""), row.get("model", ""), make_only)
            if class_key is None:
                skipped_unlabeled += 1
                continue
            image_file = row["image_file"].strip()
            if not (source_dir / image_file).is_file():
                logger.warning("listed image not found, skipping: %s", image_file)
                skipped_missing_file += 1
                continue
            by_class_key.setdefault(class_key, []).append((image_file, row.get("make", "")))

    return by_class_key, skipped_unlabeled, skipped_missing_file


def materialize_dataset(
    csv_paths: list[Path],
    source_dir: Path,
    output_dir: Path,
    min_images_per_class: int = 2,
    make_only: bool = False,
    merge: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy each labeled image into output_dir/<class_name>/.

    Fresh mode (merge=False): wipes output_dir first, class names are the
    normalized `make_model` (or `make`) key directly, and classes below
    min_images_per_class are dropped (need at least 2 images for a
    train/val split).

    Merge mode (merge=True): output_dir is never wiped or overwritten;
    class directory names are resolved via resolve_class_directory_name
    (reuse existing casing, Title-Case new classes); min_images_per_class
    is not applied — every row the caller asked to add is added, since
    dropping would silently under-deliver on an explicit "add this data"
    request. Report sparse classes to the caller via the log instead.

    dry_run=True performs every check and logs the plan without copying
    or creating anything (safe against real external datasets).

    Returns the final per-class image counts (or the *planned* counts,
    under dry_run).

    Raises:
        FileNotFoundError: if any csv_path or source_dir doesn't exist.
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source image directory not found: {source_dir}")

    by_class_key, skipped_unlabeled, skipped_missing_file = _collect_labeled_rows(
        csv_paths, source_dir, make_only
    )

    sparse_classes: dict[str, list[tuple[str, str]]] = {}
    if not merge:
        sparse_classes = {
            key: rows for key, rows in by_class_key.items() if len(rows) < min_images_per_class
        }
        for key, rows in sparse_classes.items():
            logger.warning(
                "dropping class %r: only %d image(s) (< --min-images-per-class %d)",
                key,
                len(rows),
                min_images_per_class,
            )
            del by_class_key[key]
    else:
        for key, rows in by_class_key.items():
            if len(rows) < min_images_per_class:
                logger.warning(
                    "class %r has only %d image(s) — too few for a train/val split, "
                    "but adding as requested",
                    key,
                    len(rows),
                )

    if not merge and not dry_run:
        shutil.rmtree(output_dir, ignore_errors=True)

    counts: dict[str, int] = {}
    for class_key, rows in by_class_key.items():
        if merge:
            directory_name = resolve_class_directory_name(class_key, rows[0][1], output_dir)
        else:
            directory_name = class_key
        class_dir = output_dir / directory_name

        added = 0
        for image_file, _raw_make in rows:
            destination = class_dir / image_file
            if merge and destination.exists():
                continue  # idempotent re-run: don't re-copy/count what's already there
            added += 1
            if not dry_run:
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_dir / image_file, destination)

        counts[directory_name] = counts.get(directory_name, 0) + added

    logger.info(
        "%s%d image(s) across %d class(es) -> %s "
        "(skipped: %d unlabeled/not_visible, %d missing file%s)",
        "[DRY RUN] would materialize " if dry_run else "materialized ",
        sum(counts.values()),
        len(counts),
        output_dir,
        skipped_unlabeled,
        skipped_missing_file,
        f", {len(sparse_classes)} class(es) too sparse (dropped)" if sparse_classes else "",
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csv_paths", type=Path, nargs="+", help="One or more label CSVs (see module docstring)"
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--make-only",
        action="store_true",
        help="Class = make only, ignoring the model column (rows with a blank model are kept).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Add into an existing ImageFolder directory instead of wiping --output-dir; "
        "matches existing class folders case-insensitively and never deletes anything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the plan without touching disk.",
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=2,
        help="Fresh mode only: drop classes with fewer images than this "
        "(need at least 2 for a train/val split). Ignored with --merge.",
    )
    args = parser.parse_args()

    counts = materialize_dataset(
        args.csv_paths,
        args.source_dir,
        args.output_dir,
        min_images_per_class=args.min_images_per_class,
        make_only=args.make_only,
        merge=args.merge,
        dry_run=args.dry_run,
    )
    for class_name in sorted(counts):
        print(f"  {class_name}: {counts[class_name]}")


if __name__ == "__main__":
    main()
