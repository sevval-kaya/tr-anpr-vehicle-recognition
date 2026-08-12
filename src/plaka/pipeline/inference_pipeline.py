"""End-to-end single-frame inference, tying together detection, OCR, and
classification (docs/architecture.md, "Uçtan Uca İşlem Hattı", steps 1-5).

Temporal voting across video frames (step 6) is deliberately out of scope:
this class processes one frame at a time and returns a FrameResult; a
separate tracker-aware aggregator will consume a stream of these once
video evaluation data exists (roadmap stage 3+).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from plaka.detection.plate_detector import RawPlateBox
from plaka.ocr.preprocessing import enhance_plate_crop
from plaka.pipeline.protocols import (
    PlateDetectorProtocol,
    PlateOcrProtocol,
    VehicleClassifierProtocol,
    VehicleDetectorProtocol,
)
from plaka.pipeline.schemas import BoundingBox, FrameResult, PlateReading, VehicleDetection
from plaka.utils.logging import get_logger
from plaka.validation.plate_format import TurkishPlateValidator

logger = get_logger(__name__)

# A plate is only attributed to a vehicle if at least this fraction of the
# plate box falls inside the vehicle box; below that it's dropped rather
# than guess-assigned in a multi-vehicle scene.
MIN_PLATE_CONTAINMENT_RATIO = 0.5


class InferencePipeline:
    """Runs vehicle detection, plate detection+OCR, and make/model
    classification on a single frame and assembles the combined result.
    """

    def __init__(
        self,
        vehicle_detector: VehicleDetectorProtocol,
        plate_detector: PlateDetectorProtocol,
        plate_ocr: PlateOcrProtocol,
        vehicle_classifier: VehicleClassifierProtocol,
        plate_validator: TurkishPlateValidator | None = None,
    ) -> None:
        self._vehicle_detector = vehicle_detector
        self._plate_detector = plate_detector
        self._plate_ocr = plate_ocr
        self._vehicle_classifier = vehicle_classifier
        self._plate_validator = plate_validator or TurkishPlateValidator()

    def process_frame(self, frame_bgr: NDArray[np.uint8], frame_index: int = 0) -> FrameResult:
        """Process one BGR frame (as read by OpenCV) end to end."""
        raw_vehicles = self._vehicle_detector.detect(frame_bgr)
        raw_plates = self._plate_detector.detect(frame_bgr)

        vehicles: list[VehicleDetection] = []
        for raw_vehicle in raw_vehicles:
            crop = _crop(frame_bgr, raw_vehicle.box)
            if crop.size == 0:
                logger.warning("empty vehicle crop at %s, skipping", raw_vehicle.box)
                continue

            make_model = self._vehicle_classifier.predict(crop)
            plate_reading = self._read_matching_plate(raw_vehicle.box, raw_plates, frame_bgr)

            vehicles.append(
                VehicleDetection(
                    box=raw_vehicle.box,
                    detection_confidence=raw_vehicle.confidence,
                    plate=plate_reading,
                    make_model=make_model,
                )
            )

        return FrameResult(frame_index=frame_index, vehicles=vehicles)

    def _read_matching_plate(
        self,
        vehicle_box: BoundingBox,
        raw_plates: list[RawPlateBox],
        frame_bgr: NDArray[np.uint8],
    ) -> PlateReading | None:
        if not raw_plates:
            return None

        best_plate = max(raw_plates, key=lambda p: vehicle_box.containment_ratio(p.box))
        if vehicle_box.containment_ratio(best_plate.box) < MIN_PLATE_CONTAINMENT_RATIO:
            return None

        plate_crop = _crop(frame_bgr, best_plate.box)
        if plate_crop.size == 0:
            return None

        enhanced_crop = enhance_plate_crop(plate_crop)
        ocr_reading = self._plate_ocr.read(enhanced_crop)
        validation = self._plate_validator.validate(ocr_reading.raw_text)

        return PlateReading(
            box=best_plate.box,
            raw_text=ocr_reading.raw_text,
            normalized_text=validation.normalized if validation.is_valid else None,
            is_format_valid=validation.is_valid,
            detection_confidence=best_plate.confidence,
            ocr_confidence=ocr_reading.confidence,
        )


def _crop(frame_bgr: NDArray[np.uint8], box: BoundingBox) -> NDArray[np.uint8]:
    return frame_bgr[int(box.y_min) : int(box.y_max), int(box.x_min) : int(box.x_max)]
