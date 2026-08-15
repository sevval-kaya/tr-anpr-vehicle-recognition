#!/usr/bin/env python
"""Per-class diagnostic for a brand-only classifier: run labeled Turkey
images back through a trained checkpoint and report predicted vs. true
brand.

Aggregate val_top1/top5 (printed by train_classifier.py) hides per-class
behavior and, worse, is computed over an internal split of the *training*
pool — it says nothing about images the model never had a chance to see.
By default this script evaluates data/external/vehicle_labels_pilots/
vehicle_labels_TEST_holdout.csv, the fixed set of Turkey images
deliberately excluded from every training run so results here reflect
real generalization, not memorization (see docs/decisions.md #25/#26).

    python scripts/eval_brand_classifier.py                                   # held-out test set
    python scripts/eval_brand_classifier.py --csv-dir data/external/vehicle_labels_pilots --pattern "vehicle_labels_pilot*.csv"  # old behavior: ALL labeled images (mixes in training data — memorization risk)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from plaka.classification.vehicle_classifier import VehicleClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = REPO_ROOT / "data" / "external" / "vehicle_labels_pilots"
SOURCE_DIR = REPO_ROOT / "data" / "external" / "user_plates" / "images"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "vehicle_classifier_brand_v2"
SKIP_VISIBILITY = frozenset({"not_visible"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--pattern", default="vehicle_labels_TEST_holdout.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    classifier = VehicleClassifier(
        weights_path=args.model_dir / "best.pt",
        class_names_path=args.model_dir / "classes.txt",
        architecture="efficientnet_b0",
        device=args.device,
    )

    rows: list[dict[str, str]] = []
    for csv_path in sorted(args.csv_dir.glob(args.pattern)):
        rows.extend(csv.DictReader(csv_path.open(encoding="utf-8")))

    if not rows:
        raise SystemExit(f"no rows found for {args.csv_dir}/{args.pattern}")

    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for row in rows:
        make = row.get("make", "").strip().lower()
        if not make or row.get("visibility", "").strip().lower() in SKIP_VISIBILITY:
            continue

        image = cv2.imread(str(SOURCE_DIR / row["image_file"]))
        if image is None:
            print(f"  could not read {row['image_file']}, skipping")
            continue

        prediction = classifier.predict(image, top_k=3)
        top1 = (prediction.top_1 or "").lower()
        correct = top1 == make

        per_class_total[make] = per_class_total.get(make, 0) + 1
        per_class_correct[make] = per_class_correct.get(make, 0) + int(correct)

        top3 = ", ".join(
            f"{label} ({conf:.2f})"
            for label, conf in zip(prediction.ranked_labels, prediction.ranked_confidences, strict=False)
        )
        marker = "OK  " if correct else "MISS"
        print(f"[{marker}] {row['image_file']:10s} true={make:12s} pred_top3=[{top3}]")

    print("\n--- per-brand accuracy ---")
    for make in sorted(per_class_total):
        correct = per_class_correct[make]
        total = per_class_total[make]
        print(f"  {make:12s} {correct}/{total}")

    total_correct = sum(per_class_correct.values())
    total_count = sum(per_class_total.values())
    print(f"\noverall: {total_correct}/{total_count} ({total_correct / total_count:.1%})")


if __name__ == "__main__":
    main()
