import numpy as np

from plaka.pipeline.schemas import (
    BoundingBox,
    FrameResult,
    MakeModelPrediction,
    PlateReading,
    VehicleDetection,
)
from plaka.pipeline.visualization import annotate_frame


def _frame() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_returns_a_copy_not_a_mutated_input() -> None:
    frame = _frame()
    original = frame.copy()
    result = FrameResult(frame_index=0, vehicles=[])

    annotated = annotate_frame(frame, result)

    assert annotated is not frame
    assert np.array_equal(frame, original)  # input untouched


def test_empty_result_returns_frame_shaped_output() -> None:
    frame = _frame()
    result = FrameResult(frame_index=0, vehicles=[])
    annotated = annotate_frame(frame, result)
    assert annotated.shape == frame.shape


def test_draws_something_for_a_vehicle_with_plate_and_make_model() -> None:
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="car",
        detection_confidence=0.9,
        plate=PlateReading(
            box=BoundingBox(x_min=20, y_min=80, x_max=80, y_max=95),
            raw_text="34AB123",
            normalized_text="34 AB 123",
            is_format_valid=True,
            detection_confidence=0.9,
            ocr_confidence=0.95,
        ),
        make_model=MakeModelPrediction(ranked_labels=["renault_clio"], ranked_confidences=[0.8]),
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])

    annotated = annotate_frame(frame, result)

    assert not np.array_equal(annotated, frame)  # something was drawn


def test_low_confidence_make_model_does_not_crash_and_still_draws() -> None:
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="car",
        detection_confidence=0.9,
        plate=None,
        make_model=MakeModelPrediction(ranked_labels=["ford_mustang_2001"], ranked_confidences=[0.05]),
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])

    annotated = annotate_frame(frame, result, low_confidence_threshold=0.3)

    assert not np.array_equal(annotated, frame)


def test_vehicle_with_no_plate_or_make_model_still_draws_box() -> None:
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="truck",
        detection_confidence=0.9,
        plate=None,
        make_model=None,
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])

    annotated = annotate_frame(frame, result)

    assert not np.array_equal(annotated, frame)


def test_speed_is_drawn_below_the_box_when_set() -> None:
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="car",
        detection_confidence=0.9,
        plate=None,
        estimated_speed_kmh=42.0,
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])

    without_speed = annotate_frame(
        frame,
        FrameResult(
            frame_index=0,
            vehicles=[vehicle.model_copy(update={"estimated_speed_kmh": None})],
        ),
    )
    with_speed = annotate_frame(frame, result)

    assert not np.array_equal(with_speed, without_speed)  # the speed label adds pixels


def test_speed_label_is_not_drawn_when_none() -> None:
    # Photo-mode vehicles always have estimated_speed_kmh=None (no motion
    # info in a single frame) — must not crash or draw a bogus label.
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="car",
        detection_confidence=0.9,
        plate=None,
        estimated_speed_kmh=None,
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])
    annotated = annotate_frame(frame, result)
    assert not np.array_equal(annotated, frame)  # box/type label still drawn


def test_speed_over_limit_uses_a_different_color_than_under_limit() -> None:
    # Same estimated speed (80 km/h) both times — only the configured limit
    # changes whether it's flagged as exceeded (red + "!" suffix) or not
    # (plain white label).
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="car",
        detection_confidence=0.9,
        plate=None,
        estimated_speed_kmh=80.0,
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])

    under_limit = annotate_frame(frame, result, speed_limit_kmh=100.0)
    over_limit = annotate_frame(frame, result, speed_limit_kmh=50.0)
    assert not np.array_equal(under_limit, over_limit)


def test_vehicle_type_is_shown_when_classification_disabled() -> None:
    # Default scope (docs/decisions.md #29): no classifier wired in, so
    # make_model is always None, but the box must still get a type label.
    frame = _frame()
    vehicle = VehicleDetection(
        box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
        vehicle_type="motorcycle",
        detection_confidence=0.9,
        plate=None,
        make_model=None,
    )
    result = FrameResult(frame_index=0, vehicles=[vehicle])

    annotated = annotate_frame(frame, result)

    assert not np.array_equal(annotated, frame)
