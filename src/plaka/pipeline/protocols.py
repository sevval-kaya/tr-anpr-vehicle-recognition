"""Structural interfaces for pipeline stage dependencies.

InferencePipeline depends on these Protocols rather than the concrete
VehicleDetector/PlateDetector/PlateOcr/VehicleClassifier classes, so it can
be tested with lightweight fakes (no torch/ultralytics/paddleocr needed) and
so any of those backends can be swapped without touching orchestration code.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from plaka.detection.plate_detector import RawPlateBox
from plaka.detection.vehicle_detector import RawDetection
from plaka.ocr.plate_ocr import OcrReading
from plaka.pipeline.schemas import MakeModelPrediction


class VehicleDetectorProtocol(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> list[RawDetection]: ...


class PlateDetectorProtocol(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> list[RawPlateBox]: ...


class PlateOcrProtocol(Protocol):
    def read(self, plate_crop_bgr: NDArray[np.uint8]) -> OcrReading: ...


class VehicleClassifierProtocol(Protocol):
    def predict(
        self, vehicle_crop_bgr: NDArray[np.uint8], top_k: int = 5
    ) -> MakeModelPrediction: ...
