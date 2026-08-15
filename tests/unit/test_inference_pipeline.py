from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from plaka.detection.plate_detector import RawPlateBox
from plaka.detection.vehicle_detector import RawDetection
from plaka.ocr.plate_ocr import OcrReading
from plaka.pipeline.inference_pipeline import PLATE_CROP_PADDING_RATIO, InferencePipeline, _crop
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
    assert vehicle.vehicle_type == "car"
    assert vehicle.make_model is not None
    assert vehicle.make_model.top_1 == "renault_clio"
    assert vehicle.plate is not None
    assert vehicle.plate.is_format_valid is True
    assert vehicle.plate.normalized_text == "34 AB 123"


def test_process_frame_without_classifier_leaves_make_model_none() -> None:
    # Default scope (docs/decisions.md #29): no vehicle_classifier passed in
    # -> classification is skipped entirely, vehicle_type still comes through.
    vehicle_box = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200)
    plate_box = BoundingBox(x_min=50, y_min=150, x_max=150, y_max=190)

    pipeline = InferencePipeline(
        vehicle_detector=FakeVehicleDetector([RawDetection(vehicle_box, 0.9, "truck")]),
        plate_detector=FakePlateDetector([RawPlateBox(plate_box, 0.95)]),
        plate_ocr=FakePlateOcr(OcrReading(raw_text="34AB123", confidence=0.88)),
    )

    result = pipeline.process_frame(_frame())

    assert len(result.vehicles) == 1
    vehicle = result.vehicles[0]
    assert vehicle.vehicle_type == "truck"
    assert vehicle.make_model is None
    assert vehicle.plate is not None
    assert vehicle.plate.is_format_valid is True


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


class TestCrop:
    def test_zero_padding_crops_exactly_the_box(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = BoundingBox(x_min=50, y_min=60, x_max=150, y_max=100)
        crop = _crop(frame, box)
        assert crop.shape[:2] == (40, 100)  # (height, width)

    def test_padding_expands_the_crop_by_the_given_ratio(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = BoundingBox(x_min=50, y_min=60, x_max=150, y_max=100)  # 100x40
        crop = _crop(frame, box, padding_ratio=0.15)
        # pad_x = 100*0.15 = 15 each side -> width 130; pad_y = 40*0.15 = 6 each side -> height 52
        assert crop.shape[:2] == (52, 130)

    def test_padding_is_clamped_to_frame_bounds(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box = BoundingBox(x_min=0, y_min=0, x_max=20, y_max=20)
        crop = _crop(frame, box, padding_ratio=1.0)  # would go to -20..40 without clamping
        assert crop.shape[:2] == (40, 40)


def test_plate_ocr_receives_a_padded_crop_not_the_tight_detector_box() -> None:
    # Regression test: a zero-padding plate crop was found to make
    # PaddleOCR's text detector return nothing at all on an otherwise
    # perfectly legible plate (see docs/decisions.md) — process_frame must
    # pad the plate crop before handing it to plate_ocr.read().
    vehicle_box = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200)
    plate_box = BoundingBox(x_min=50, y_min=150, x_max=150, y_max=190)  # 100x40

    plate_ocr = FakePlateOcr(OcrReading(raw_text="34AB123", confidence=0.88))
    pipeline = InferencePipeline(
        vehicle_detector=FakeVehicleDetector([RawDetection(vehicle_box, 0.9, "car")]),
        plate_detector=FakePlateDetector([RawPlateBox(plate_box, 0.95)]),
        plate_ocr=plate_ocr,
        vehicle_classifier=FakeVehicleClassifier(
            MakeModelPrediction(ranked_labels=["renault_clio"], ranked_confidences=[0.7])
        ),
    )

    pipeline.process_frame(_frame())

    assert len(plate_ocr.calls) == 1
    tight_height, tight_width = 40, 100
    padded_height, padded_width = plate_ocr.calls[0].shape[:2]
    assert padded_width > tight_width
    assert padded_height > tight_height
    assert padded_width == round(tight_width * (1 + 2 * PLATE_CROP_PADDING_RATIO))
