"""Cropping plate regions straight from a raw YOLO label row.

Separate from `plaka.pipeline.inference_pipeline`'s `_crop` helper, which
operates on a pixel-space `BoundingBox` produced by the detector at
inference time: this one starts from a normalized YOLO label line (the
format `data/external/*` datasets ship), for tools that need to go
directly from an on-disk label file to a crop without running detection
(see scripts/label_plates_with_claude.py).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Small margin around the tight bbox so characters aren't clipped by an
# off-by-a-few-pixels label, and so OCR sees a bit of border context.
DEFAULT_PADDING_RATIO = 0.08


def parse_yolo_label_line(line: str) -> tuple[int, float, float, float, float]:
    """Parse one `class x_center y_center width height` row (values normalized 0-1).

    Raises:
        ValueError: if the row doesn't have exactly 5 fields.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 whitespace-separated fields, got: {line!r}")
    class_id = int(parts[0])
    x_center, y_center, width, height = (float(v) for v in parts[1:])
    return class_id, x_center, y_center, width, height


def largest_box_line(label_text: str) -> str | None:
    """Pick the row with the largest bbox area, for label files with more
    than one row — the plate is expected to be the dominant object, so the
    largest box is the safest single choice when multiple rows exist.

    Returns None if `label_text` has no non-blank rows.
    """
    best_line: str | None = None
    best_area = -1.0
    for line in label_text.splitlines():
        if not line.strip():
            continue
        _class_id, _x, _y, width, height = parse_yolo_label_line(line)
        area = width * height
        if area > best_area:
            best_area = area
            best_line = line
    return best_line


def crop_plate_from_yolo_box(
    image_bgr: NDArray[np.uint8],
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
) -> NDArray[np.uint8]:
    """Denormalize a YOLO bbox against `image_bgr`'s dimensions and crop it,
    with a small padding margin, clamped to the image bounds.
    """
    img_h, img_w = image_bgr.shape[:2]

    box_w = width * img_w
    box_h = height * img_h
    center_x = x_center * img_w
    center_y = y_center * img_h

    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio

    x_min = max(0, int(round(center_x - box_w / 2 - pad_w)))
    y_min = max(0, int(round(center_y - box_h / 2 - pad_h)))
    x_max = min(img_w, int(round(center_x + box_w / 2 + pad_w)))
    y_max = min(img_h, int(round(center_y + box_h / 2 + pad_h)))

    return image_bgr[y_min:y_max, x_min:x_max]
