"""Vehicle detection stage of the pipeline.

Baseline decision (see docs/decisions.md): vehicle detection does not need
custom training to get started. COCO already defines car/motorcycle/bus/
truck classes, so a stock COCO-pretrained YOLO checkpoint is used as-is for
the stage-2 baseline; a Turkish-traffic fine-tuned checkpoint can replace it
later if COCO-pretrained recall proves insufficient in zorlu senaryo testing.
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

# COCO class ids restricted to vehicle categories relevant to this project.
COCO_VEHICLE_CLASS_IDS: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True, slots=True)
class RawDetection:
    box: BoundingBox
    confidence: float
    class_name: str


class VehicleDetector:
    """Detects vehicles in a frame using an Ultralytics YOLO checkpoint.

    Model weights are loaded lazily on first `detect()` call so that
    importing this module never requires `ultralytics` to be installed
    (it's an optional dependency group; see pyproject.toml).
    """

    def __init__(
        self,
        weights_path: str | Path = "yolo11n.pt",
        confidence_threshold: float = 0.4,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._confidence_threshold = confidence_threshold
        self._model: YOLO | None = None

    def _ensure_model_loaded(self) -> YOLO:
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore[attr-defined]
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is required for VehicleDetector; install with "
                    "`pip install -e '.[detection]'`"
                ) from exc
            self._model = YOLO(str(self._weights_path))
        return self._model

    def detect(self, frame: NDArray[np.uint8]) -> list[RawDetection]:
        """Run vehicle detection on a single BGR frame (as read by OpenCV).

        Returns detections for COCO vehicle classes only; all other COCO
        classes the checkpoint may predict are filtered out.
        """
        model = self._ensure_model_loaded()
        # ultralytics' predict() return-type stub is a broad union that
        # doesn't reflect the single-frame, non-streaming call shape used
        # here; narrow it at this one boundary rather than fighting the
        # stub throughout the method.
        results = cast(
            "list[Any]", model.predict(frame, conf=self._confidence_threshold, verbose=False)
        )

        detections: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                if class_id not in COCO_VEHICLE_CLASS_IDS:
                    continue
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                detections.append(
                    RawDetection(
                        box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                        confidence=float(box.conf.item()),
                        class_name=COCO_VEHICLE_CLASS_IDS[class_id],
                    )
                )
        return detections
