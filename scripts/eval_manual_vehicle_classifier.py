#!/usr/bin/env python
"""Evaluate a vehicle brand classifier checkpoint against the held-out
test set from scripts/prepare_manual_vehicle_labels.py
(data/processed/vehicle_labels_manual_split/test_holdout.csv by default)
— images never used for training or for the internal train/val split
inside train_classifier.py, so this is the honest generalization number
(see docs/decisions.md #27).

    python scripts/eval_manual_vehicle_classifier.py --model-dir models/vehicle_classifier_manual_frozen
    python scripts/eval_manual_vehicle_classifier.py --model-dir models/vehicle_classifier_manual_partial

Reports overall top-1 accuracy, per-class accuracy, and the most common
confused (true, predicted) brand pairs among the misses.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from plaka.classification.vehicle_classifier import VehicleClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_CSV = REPO_ROOT / "data" / "processed" / "vehicle_labels_manual_split" / "test_holdout.csv"
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "external" / "user_plates" / "images"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--architecture", default="efficientnet_b0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    classifier = VehicleClassifier(
        weights_path=args.model_dir / "best.pt",
        class_names_path=args.model_dir / "classes.txt",
        architecture=args.architecture,
        device=args.device,
    )

    rows = list(csv.DictReader(args.test_csv.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {args.test_csv}")

    per_class_correct: Counter[str] = Counter()
    per_class_total: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()

    print(f"=== {args.model_dir.name} on {len(rows)} held-out image(s) ===\n")
    for row in rows:
        true_make = row["make"].strip().lower()
        image = cv2.imread(str(args.source_dir / row["image_file"]))
        if image is None:
            print(f"  could not read {row['image_file']}, skipping")
            continue

        prediction = classifier.predict(image, top_k=3)
        top1 = (prediction.top_1 or "").lower()
        correct = top1 == true_make

        per_class_total[true_make] += 1
        per_class_correct[true_make] += int(correct)
        if not correct:
            confusions[(true_make, top1)] += 1

        top3 = ", ".join(
            f"{label} ({conf:.2f})"
            for label, conf in zip(prediction.ranked_labels, prediction.ranked_confidences, strict=False)
        )
        marker = "OK  " if correct else "MISS"
        print(f"[{marker}] {row['image_file']:10s} true={true_make:14s} pred_top3=[{top3}]")

    print("\n--- per-brand accuracy ---")
    for make in sorted(per_class_total, key=lambda m: -per_class_total[m]):
        print(f"  {make:15s} {per_class_correct[make]}/{per_class_total[make]}")

    total_correct = sum(per_class_correct.values())
    total_count = sum(per_class_total.values())
    print(f"\noverall: {total_correct}/{total_count} ({total_correct / total_count:.1%})")

    if confusions:
        print("\n--- most confused (true -> predicted) pairs ---")
        for (true_make, pred_make), count in confusions.most_common():
            print(f"  {true_make} -> {pred_make}: {count}")


if __name__ == "__main__":
    main()
