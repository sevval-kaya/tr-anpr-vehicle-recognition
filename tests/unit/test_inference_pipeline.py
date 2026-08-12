from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plaka.detection.plate_detector import RawPlateBox
from plaka.detection.vehicle_detector import RawDetection
from plaka.ocr.plate_ocr import OcrReading
from plaka.pipeline.inference_pipeline import InferencePipeline
from plaka.pipeline.schemas import BoundingBox, MakeModelPrediction


@dataclass
class FakeVehicleDetector:
    detections: list[RawDetection]

    def detect(self, frame: NDArray[np.uint8]) -> list[RawDetection]:
        return self.detections


@dataclass
class FakePlateDetector:
    detections: list[RawPlateBox]

    def detect(self, frame: NDArray[np.uint8]) -> list[RawPlateBox]:
        return self.detections


@dataclass
class FakePlateOcr:
    reading: OcrReading
    calls: list[NDArray[np.uint8]] = field(default_factory=list)

    def read(self, plate_crop_bgr: NDArray[np.uint8]) -> OcrReading:
        self.calls.append(plate_crop_bgr)
        return self.reading


@dataclass
class FakeVehicleClassifier:
    prediction: MakeModelPrediction

    def predict(self, vehicle_crop_bgr: NDArray[np.uint8], top_k: int = 5) -> MakeModelPrediction:
        return self.prediction


def _frame() -> NDArray[np.uint8]:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_process_frame_attaches_plate_to_containing_vehicle() -> None:
    vehicle_box = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200)
    plate_box = BoundingBox(x_min=50, y_min=150, x_max=150, y_max=190)

    pipeline = InferencePipeline(
        vehicle_detector=FakeVehicleDetector([RawDetection(vehicle_box, 0.9, "car")]),
        plate_detector=FakePlateDetector([RawPlateBox(plate_box, 0.95)]),
        plate_ocr=FakePlateOcr(OcrReading(raw_text="34AB123", confidence=0.88)),
        vehicle_classifier=FakeVehicleClassifier(
            MakeModelPrediction(ranked_labels=["renault_clio"], ranked_confidences=[0.7])
        ),
    )

    result = pipeline.process_frame(_frame(), frame_index=3)

    assert result.frame_index == 3
    assert len(result.vehicles) == 1
    vehicle = result.vehicles[0]
    assert vehicle.make_model is not None
    assert vehicle.make_model.top_1 == "renault_clio"
    assert vehicle.plate is not None
    assert vehicle.plate.is_format_valid is True
    assert vehicle.plate.normalized_text == "34 AB 123"


def test_process_frame_drops_plate_not_contained_in_any_vehicle() -> None:
    vehicle_box = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
    far_away_plate = BoundingBox(x_min=150, y_min=150, x_max=190, y_max=180)

    pipeline = InferencePipeline(
        vehicle_detector=FakeVehicleDetector([RawDetection(vehicle_box, 0.9, "car")]),
        plate_detector=FakePlateDetector([RawPlateBox(far_away_plate, 0.95)]),
        plate_ocr=FakePlateOcr(OcrReading(raw_text="34AB123", confidence=0.88)),
        vehicle_classifier=FakeVehicleClassifier(
            MakeModelPrediction(ranked_labels=["fiat_egea"], ranked_confidences=[0.6])
        ),
    )

    result = pipeline.process_frame(_frame())

    assert len(result.vehicles) == 1
    assert result.vehicles[0].plate is None


def test_process_frame_with_no_vehicles_returns_empty_result() -> None:
    pipeline = InferencePipeline(
        vehicle_detector=FakeVehicleDetector([]),
        plate_detector=FakePlateDetector([]),
        plate_ocr=FakePlateOcr(OcrReading(raw_text="", confidence=0.0)),
        vehicle_classifier=FakeVehicleClassifier(
            MakeModelPrediction(ranked_labels=[], ranked_confidences=[])
        ),
    )

    result = pipeline.process_frame(_frame())

    assert result.vehicles == []


def test_invalid_ocr_text_yields_no_normalized_text() -> None:
    vehicle_box = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200)
    plate_box = BoundingBox(x_min=50, y_min=150, x_max=150, y_max=190)

    pipeline = InferencePipeline(
        vehicle_detector=FakeVehicleDetector([RawDetection(vehicle_box, 0.9, "car")]),
        plate_detector=FakePlateDetector([RawPlateBox(plate_box, 0.95)]),
        plate_ocr=FakePlateOcr(OcrReading(raw_text="GARBLED", confidence=0.3)),
        vehicle_classifier=FakeVehicleClassifier(
            MakeModelPrediction(ranked_labels=["renault_clio"], ranked_confidences=[0.7])
        ),
    )

    result = pipeline.process_frame(_frame())

    plate = result.vehicles[0].plate
    assert plate is not None
    assert plate.is_format_valid is False
    assert plate.normalized_text is None
