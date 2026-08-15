#!/usr/bin/env python
"""Bootstrap OCR text labels for data/external/user_plates/ (1,955 bbox-only
images, no ground-truth plate text) using Claude vision as a semi-automated
labeler: crop each image's plate with its existing YOLO bbox, transcribe the
crop, and validate the transcription with TurkishPlateValidator.

Pilot-first workflow (see docs/decisions.md): run a small random sample,
manually spot-check the crops + predicted text, and only pass --full once
the pilot's accuracy looks good.

    python scripts/label_plates_with_claude.py --pilot 150
    python scripts/label_plates_with_claude.py --full

Requires ANTHROPIC_API_KEY in the environment and the `labeling` extra
(`pip install -e '.[labeling]'`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path
from typing import Any

import cv2

from plaka.data.plate_crop import crop_plate_from_yolo_box, largest_box_line, parse_yolo_label_line
from plaka.data.yolo_dataset import YoloExample, find_yolo_examples, long_path
from plaka.ocr.claude_labeler import UNREADABLE_MARKER, transcribe_plate_crop
from plaka.utils.logging import get_logger
from plaka.validation.plate_format import TurkishPlateValidator

logger = get_logger(__name__)

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "data" / "external" / "user_plates"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "ocr_labels"
PROGRESS_INTERVAL = 25


def _select_examples(source: Path, sample_size: int | None, seed: int) -> list[YoloExample]:
    examples = find_yolo_examples(source)
    labeled = [example for example in examples if example.label_path.exists()]
    skipped = len(examples) - len(labeled)
    if skipped:
        logger.warning("%d image(s) with no label file skipped", skipped)

    if sample_size is not None and sample_size < len(labeled):
        return random.Random(seed).sample(labeled, sample_size)
    return labeled


def _label_one(
    example: YoloExample,
    crops_dir: Path,
    validator: TurkishPlateValidator,
    client: Any,
) -> dict[str, Any]:
    image = cv2.imread(long_path(example.image_path))
    if image is None:
        return {"image": example.image_path.name, "error": "image_read_failed"}

    label_text = example.label_path.read_text(encoding="utf-8")
    box_line = largest_box_line(label_text)
    if box_line is None:
        return {"image": example.image_path.name, "error": "no_label_rows"}

    _class_id, x_center, y_center, width, height = parse_yolo_label_line(box_line)
    crop = crop_plate_from_yolo_box(image, x_center, y_center, width, height)
    if crop.size == 0:
        return {"image": example.image_path.name, "error": "empty_crop"}

    crop_path = crops_dir / f"{example.image_path.stem}.jpg"
    cv2.imwrite(str(crop_path), crop)

    raw_text = transcribe_plate_crop(client, crop)
    validation = None if raw_text == UNREADABLE_MARKER else validator.validate(raw_text)

    return {
        "image": example.image_path.name,
        "crop": crop_path.name,
        "claude_raw_text": raw_text,
        "is_valid": bool(validation and validation.is_valid),
        "normalized_text": validation.normalized if validation else None,
        "reason": validation.reason if validation else "unreadable",
    }


def run(
    source: Path,
    output_dir: Path,
    sample_size: int | None,
    seed: int,
    concurrency: int,
    run_name: str,
) -> Path:
    import anthropic

    client = anthropic.Anthropic()

    examples = _select_examples(source, sample_size, seed)
    logger.info("labeling %d image(s) from %s", len(examples), source)

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    validator = TurkishPlateValidator()

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_label_one, example, crops_dir, validator, client): example
            for example in examples
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if count % PROGRESS_INTERVAL == 0 or count == len(examples):
                logger.info("  %d/%d", count, len(examples))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out_file:
        for result in results:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    n_valid = sum(1 for r in results if r.get("is_valid"))
    n_error = sum(1 for r in results if "error" in r)
    logger.info(
        "done: %d/%d passed format validation, %d error(s). labels -> %s, crops -> %s",
        n_valid,
        len(results),
        n_error,
        out_path,
        crops_dir,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int, metavar="N", help="Label a random sample of N images")
    mode.add_argument("--full", action="store_true", help="Label the entire dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    run(
        source=args.source,
        output_dir=args.output_dir,
        sample_size=None if args.full else args.pilot,
        seed=args.seed,
        concurrency=args.concurrency,
        run_name="full_labels" if args.full else "pilot_labels",
    )


if __name__ == "__main__":
    main()
