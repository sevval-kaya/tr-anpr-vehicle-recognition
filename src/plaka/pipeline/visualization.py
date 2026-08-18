"""Drawing a FrameResult onto its source frame — shared by the image and
video/camera inference scripts so annotation stays visually consistent
between them.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from plaka.pipeline.schemas import FrameResult

VEHICLE_BOX_COLOR = (0, 200, 0)  # green, BGR
PLATE_BOX_COLOR = (23, 10, 227)  # TR-flag red (#e30a17), BGR — matches the web UI's --red
LOW_CONFIDENCE_COLOR = (0, 165, 255)  # orange, BGR
SPEED_LABEL_COLOR = (160, 63, 26)  # brand blue (#1a3fa0), BGR — matches the web UI's --blue
SPEED_EXCEEDED_COLOR = (23, 10, 227)  # same alert red as PLATE_BOX_COLOR / the web UI
CHIP_TEXT_COLOR = (255, 255, 255)  # white, BGR

# Below this top-1 confidence, the make/model label is flagged as unreliable
# rather than presented as a plain read — VehicleClassifier's current
# checkpoint is VMMRdb-trained (US-market skew, see docs/decisions.md #13),
# so low-confidence Turkey-market predictions are the expected common case,
# not an error condition.
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.3

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOX_THICKNESS = 2
_CHIP_PADDING_X = 7
_CHIP_PADDING_Y = 5
_CHIP_GAP = 4  # gap between a box edge and the chip anchored to it


def _draw_label_chip(
    canvas: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    background_color: tuple[int, int, int],
    font_scale: float,
    anchor: str,
) -> None:
    """Draws `text` on a small filled, anti-aliased background chip rather
    than bare colored text directly on the frame — a solid backing plate
    is what actually keeps a label legible over a busy background and is
    the main thing separating a "labeled" look from a "plain text floating
    on the photo" one (see docs/decisions.md #42 — added after that flat
    text style read as too large/unpolished at typical dashcam resolution).

    `anchor` is `"above"` (chip's bottom edge sits `_CHIP_GAP`px above y —
    used for the vehicle/plate label above the box) or `"below"` (chip's
    top edge sits `_CHIP_GAP`px below y — used for the speed label under
    the box). `x` is always the chip's left edge.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, font_scale, 1)
    chip_w = text_w + 2 * _CHIP_PADDING_X
    chip_h = text_h + baseline + 2 * _CHIP_PADDING_Y

    if anchor == "above":
        chip_y2 = y - _CHIP_GAP
        chip_y1 = chip_y2 - chip_h
    else:
        chip_y1 = y + _CHIP_GAP
        chip_y2 = chip_y1 + chip_h

    height, width = canvas.shape[:2]
    chip_y1 = max(0, min(chip_y1, height - 1))
    chip_y2 = max(0, min(chip_y2, height - 1))
    chip_x1 = max(0, min(x, width - 1))
    chip_x2 = max(0, min(x + chip_w, width - 1))

    cv2.rectangle(canvas, (chip_x1, chip_y1), (chip_x2, chip_y2), background_color, -1, cv2.LINE_AA)
    text_origin = (chip_x1 + _CHIP_PADDING_X, chip_y2 - _CHIP_PADDING_Y - baseline)
    cv2.putText(canvas, text, text_origin, _FONT, font_scale, CHIP_TEXT_COLOR, 1, cv2.LINE_AA)


def annotate_frame(
    image_bgr: NDArray[np.uint8],
    result: FrameResult,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    speed_limit_kmh: float | None = None,
) -> NDArray[np.uint8]:
    """Return a copy of `image_bgr` with vehicle/plate boxes and read
    text/make-model labels drawn on. Does not mutate `image_bgr`.

    `speed_limit_kmh`, when given, only affects color (red once a vehicle's
    `estimated_speed_kmh` exceeds it) — the uncalibrated speed estimate
    itself (docs/decisions.md #42) is drawn below the vehicle box whenever
    it's set, regardless of whether a limit was passed. It's never set for
    single-frame photo results (no motion to estimate from), so this is a
    no-op there.
    """
    canvas = image_bgr.copy()
    for vehicle in result.vehicles:
        box = vehicle.box
        x_min, y_min, x_max, y_max = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
        cv2.rectangle(
            canvas, (x_min, y_min), (x_max, y_max), VEHICLE_BOX_COLOR, _BOX_THICKNESS, cv2.LINE_AA
        )

        label_parts = [vehicle.vehicle_type]
        label_color = VEHICLE_BOX_COLOR
        # make_model is only set when classification is explicitly re-enabled
        # (configs/pipeline.yaml classification.enabled) — see decision #29.
        if vehicle.make_model and vehicle.make_model.top_1:
            top1_confidence = vehicle.make_model.ranked_confidences[0]
            if top1_confidence < low_confidence_threshold:
                label_parts.append(f"{vehicle.make_model.top_1}? ({top1_confidence:.0%})")
                label_color = LOW_CONFIDENCE_COLOR
            else:
                label_parts.append(vehicle.make_model.top_1)

        if vehicle.plate is not None:
            plate_text = vehicle.plate.normalized_text or vehicle.plate.raw_text
            label_parts.append(plate_text or "?")
            plate_box = vehicle.plate.box
            cv2.rectangle(
                canvas,
                (int(plate_box.x_min), int(plate_box.y_min)),
                (int(plate_box.x_max), int(plate_box.y_max)),
                PLATE_BOX_COLOR,
                _BOX_THICKNESS,
                cv2.LINE_AA,
            )

        label = " | ".join(label_parts) if label_parts else "?"
        _draw_label_chip(canvas, label, x_min, y_min, label_color, font_scale=0.55, anchor="above")

        if vehicle.estimated_speed_kmh is not None:
            exceeded = (
                speed_limit_kmh is not None and vehicle.estimated_speed_kmh > speed_limit_kmh
            )
            speed_label = f"~{vehicle.estimated_speed_kmh:.0f} km/h"
            speed_color = SPEED_EXCEEDED_COLOR if exceeded else SPEED_LABEL_COLOR
            _draw_label_chip(
                canvas, speed_label, x_min, y_max, speed_color, font_scale=0.5, anchor="below"
            )
    return canvas
