#!/usr/bin/env python
"""End-to-end smoke test: run vehicle detection, plate detection+OCR, and
make/model classification on one or more real images via InferencePipeline
(configs/pipeline.yaml + detection.yaml/ocr.yaml/classification.yaml).

    python scripts/run_inference.py path/to/image.jpg
    python scripts/run_inference.py path/to/dir/ --annotate

Prints one FrameResult per image; with --annotate, also writes an annotated
copy (vehicle/plate boxes + read text + top-1 make/model) to outputs/.

For live camera or video-file input, see scripts/run_inference_video.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from plaka.pipeline.builder import REPO_ROOT, build_pipeline_from_config
from plaka.pipeline.schemas import FrameResult
from plaka.pipeline.visualization import annotate_frame
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


def _print_result(image_path: Path, result: FrameResult) -> None:
    print(f"\n{image_path.name}: {len(result.vehicles)} vehicle(s)")
    for i, vehicle in enumerate(result.vehicles, start=1):
        box = vehicle.box
        print(
            f"  [{i}] {vehicle.vehicle_type} box=({box.x_min:.0f},{box.y_min:.0f})-"
            f"({box.x_max:.0f},{box.y_max:.0f}) conf={vehicle.detection_confidence:.2f}"
        )
        if vehicle.plate is None:
            print("      plate: none matched")
        else:
            plate = vehicle.plate
            status = "VALID" if plate.is_format_valid else "invalid format"
            text = plate.normalized_text or plate.raw_text or "(empty)"
            print(
                f"      plate: {text!r} [{status}] "
                f"(det={plate.detection_confidence:.2f}, ocr={plate.ocr_confidence:.2f})"
            )
        # make_model is only populated when classification is explicitly
        # re-enabled (configs/pipeline.yaml classification.enabled) — see
        # docs/decisions.md #29.
        if vehicle.make_model is not None and vehicle.make_model.ranked_labels:
            mm = vehicle.make_model
            top3 = ", ".join(
                f"{label} ({conf:.2f})"
                for label, conf in zip(mm.ranked_labels[:3], mm.ranked_confidences[:3], strict=False)
            )
            print(f"      make/model top-3: {top3}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory of images")
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--annotate", action="store_true", help="Write annotated copies to --output-dir")
    args = parser.parse_args()

    if args.input.is_dir():
        image_paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        image_paths = [args.input]

    pipeline = build_pipeline_from_config(args.pipeline_config)

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            logger.warning("failed to read %s, skipping", image_path)
            continue

        result = pipeline.process_frame(image_bgr)
        _print_result(image_path, result)

        if args.annotate:
            output_path = args.output_dir / f"annotated_{image_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotate_frame(image_bgr, result))
            logger.info("annotated image -> %s", output_path)


if __name__ == "__main__":
    main()
