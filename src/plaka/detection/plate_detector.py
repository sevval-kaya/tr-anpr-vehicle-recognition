"""Plate region detection stage.

Unlike vehicle detection, plate detection has no free COCO-pretrained
equivalent — it requires a checkpoint fine-tuned on plate-bounding-box data
(open-source Turkish/multi-country plate datasets initially, then our own
collected data). See docs/architecture.md section 3.2 and roadmap stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from plaka.pipeline.schemas import BoundingBox

if TYPE_CHECKING:
    from ultralytics import YOLO  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class RawPlateBox:
    box: BoundingBox
    confidence: float


class PlateDetector:
    """Detects plate regions using a plate-specific YOLO checkpoint.

    Requires `weights_path` to point at a checkpoint trained on plate
    bounding-box data — there is no pretrained COCO equivalent for this
    class, unlike VehicleDetector.
    """

    def __init__(self, weights_path: str | Path, confidence_threshold: float = 0.5) -> None:
        self._weights_path = Path(weights_path)
        self._confidence_threshold = confidence_threshold
        self._model: YOLO | None = None

    def _ensure_model_loaded(self) -> YOLO:
        if self._model is None:
            if not self._weights_path.exists():
                raise FileNotFoundError(
                    f"plate detector weights not found at {self._weights_path}; "
                    "train a checkpoint first (see scripts/, roadmap stage 2)"
                )
            try:
                from ultralytics import YOLO  # type: ignore[attr-defined]
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for PlateDetector; install with "
                    "`pip install -e '.[detection]'`"
                ) from exc
            self._model = YOLO(str(self._weights_path))
        return self._model

    def detect(self, frame: NDArray[np.uint8]) -> list[RawPlateBox]:
        """Run plate detection on a single BGR frame (as read by OpenCV)."""
        model = self._ensure_model_loaded()
        # See VehicleDetector.detect for why this cast is needed.
        results = cast(
            "list[Any]", model.predict(frame, conf=self._confidence_threshold, verbose=False)
        )

        detections: list[RawPlateBox] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                detections.append(
                    RawPlateBox(
                        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                        confidence=float(box.conf.item()),
                    )
                )
        return detections
