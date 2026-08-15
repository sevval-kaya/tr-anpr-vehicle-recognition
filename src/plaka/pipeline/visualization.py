"""Drawing a FrameResult onto its source frame — shared by the image and
video/camera inference scripts so annotation stays visually consistent
between them.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from plaka.pipeline.schemas import FrameResult

VEHICLE_BOX_COLOR = (0, 255, 0)  # green, BGR
PLATE_BOX_COLOR = (0, 0, 255)  # red, BGR
LOW_CONFIDENCE_COLOR = (0, 165, 255)  # orange, BGR

# Below this top-1 confidence, the make/model label is flagged as unreliable
# rather than presented as a plain read — VehicleClassifier's current
# checkpoint is VMMRdb-trained (US-market skew, see docs/decisions.md #13),
# so low-confidence Turkey-market predictions are the expected common case,
# not an error condition.
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.3


def annotate_frame(
    image_bgr: NDArray[np.uint8],
    result: FrameResult,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
) -> NDArray[np.uint8]:
    """Return a copy of `image_bgr` with vehicle/plate boxes and read
    text/make-model labels drawn on. Does not mutate `image_bgr`.
    """
    canvas = image_bgr.copy()
    for vehicle in result.vehicles:
        box = vehicle.box
        cv2.rectangle(
            canvas,
            (int(box.x_min), int(box.y_min)),
            (int(box.x_max), int(box.y_max)),
            VEHICLE_BOX_COLOR,
            3,
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
                3,
            )

        label = " | ".join(label_parts) if label_parts else "?"
        cv2.putText(
            canvas,
            label,
            (int(box.x_min), max(0, int(box.y_min) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            label_color,
            2,
        )
    return canvas
