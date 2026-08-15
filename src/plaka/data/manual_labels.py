"""Merging, deduplicating, and splitting hand-labeled make/model CSVs
(e.g. data/external/user_plates/labels_manual/vehicle_labels_pilot*.csv —
Claude-vision-assisted manual labeling of real Turkish traffic photos, see
docs/decisions.md #21/#25/#26) into a clean train/held-out-test pair.

CSV format: `image_file,make,model,visibility`. `visibility` is one of
"clear", "partial", "not_visible". Multiple CSVs may label the same
image_file (re-labeled across separate pilot batches) — this module
reconciles those into one row per image.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LabelConflict:
    """Two source CSVs disagree on the make for the same image_file."""

    image_file: str
    entries: tuple[tuple[str, str], ...]  # (source_csv_name, make)


@dataclass(frozen=True, slots=True)
class MergedLabels:
    # image_file -> (make, model), already lowercased/stripped
    usable: dict[str, tuple[str, str]]
    excluded_not_visible: list[str]
    conflicts: list[LabelConflict]


def load_label_csvs(csv_dir: str | Path, pattern: str = "*.csv") -> list[tuple[str, dict[str, str]]]:
    """Read every CSV in csv_dir, returning (source_filename, row) pairs
    in file-then-row order (deterministic, since glob results are sorted).

    Raises:
        FileNotFoundError: if csv_dir doesn't exist.
    """
    root = Path(csv_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"label CSV directory not found: {root}")

    rows: list[tuple[str, dict[str, str]]] = []
    for csv_path in sorted(root.glob(pattern)):
        for row in csv.DictReader(csv_path.open(encoding="utf-8")):
            rows.append((csv_path.name, row))
    return rows


def merge_and_dedupe_labels(rows: list[tuple[str, dict[str, str]]]) -> MergedLabels:
    """Reconcile raw (source, row) pairs into one label per image_file.

    - Rows with visibility=="not_visible" or a blank make are excluded.
    - When the same image_file appears more than once with the *same*
      make (typical: re-labeled in a later pilot batch), one row is kept
      — preferring whichever has a non-blank model.
    - When the same image_file has *different* makes across sources,
      that's a genuine labeling contradiction: it's reported as a
      LabelConflict and excluded from `usable` rather than silently
      picking one (a wrong guess here would poison a class with a
      mislabeled example).
    """
    by_file: dict[str, list[tuple[str, str, str, str]]] = {}  # image_file -> [(source, make, model, vis)]
    for source, row in rows:
        image_file = row["image_file"].strip()
        make = row.get("make", "").strip().lower()
        model = row.get("model", "").strip().lower()
        visibility = row.get("visibility", "").strip().lower()
        by_file.setdefault(image_file, []).append((source, make, model, visibility))

    usable: dict[str, tuple[str, str]] = {}
    excluded_not_visible: list[str] = []
    conflicts: list[LabelConflict] = []

    for image_file, entries in by_file.items():
        makes = {make for _source, make, _model, _vis in entries if make}
        if len(makes) > 1:
            conflicts.append(
                LabelConflict(
                    image_file=image_file,
                    entries=tuple((source, make) for source, make, _model, _vis in entries),
                )
            )
            continue

        make = next(iter(makes), "")
        if not make:
            # Every entry had a blank make — the CSV convention is that
            # this always co-occurs with visibility=="not_visible", so a
            # blank make alone is a sufficient and simpler exclusion
            # signal than re-checking the visibility strings.
            excluded_not_visible.append(image_file)
            continue

        # All entries agree on make — reduce to one (make, model),
        # preferring whichever source gave a non-blank model.
        model = ""
        for _source, entry_make, entry_model, _entry_vis in entries:
            if entry_make == make and entry_model and not model:
                model = entry_model

        usable[image_file] = (make, model)

    return MergedLabels(usable=usable, excluded_not_visible=excluded_not_visible, conflicts=conflicts)


def stratified_holdout_split(
    usable: dict[str, tuple[str, str]],
    seed: int = 42,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], list[str]]:
    """Split into (train, test) holding out exactly one image per brand
    for every brand with at least 2 examples; brands with only 1 example
    contribute it to train only (there's nothing to hold out without
    leaving the class with zero training signal).

    Returns (train, test, single_image_brands) — the third element lists
    brands that had no held-out test example, for explicit reporting.
    """
    by_brand: dict[str, list[str]] = {}
    for image_file, (make, _model) in usable.items():
        by_brand.setdefault(make, []).append(image_file)

    rng = random.Random(seed)
    test_files: set[str] = set()
    single_image_brands: list[str] = []
    for brand, files in by_brand.items():
        if len(files) >= 2:
            test_files.add(rng.choice(sorted(files)))
        else:
            single_image_brands.append(brand)

    train = {f: v for f, v in usable.items() if f not in test_files}
    test = {f: v for f, v in usable.items() if f in test_files}
    return train, test, sorted(single_image_brands)
