import pytest

from plaka.pipeline.schemas import BoundingBox, MakeModelPrediction


def test_bounding_box_rejects_degenerate_coords() -> None:
    with pytest.raises(ValueError):
        BoundingBox(x_min=10, y_min=10, x_max=5, y_max=20)


def test_iou_of_identical_boxes_is_one() -> None:
    box = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    a = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    b = BoundingBox(x_min=100, y_min=100, x_max=110, y_max=110)
    assert a.iou(b) == 0.0


def test_containment_ratio_full_containment_is_one() -> None:
    vehicle = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200)
    plate = BoundingBox(x_min=50, y_min=150, x_max=150, y_max=190)
    assert vehicle.containment_ratio(plate) == pytest.approx(1.0)


def test_containment_ratio_low_iou_but_full_containment() -> None:
    # A small plate fully inside a much larger vehicle box scores a low
    # IoU (union dominated by the vehicle) but full containment — this is
    # exactly the case containment_ratio exists to handle correctly.
    vehicle = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=200)
    plate = BoundingBox(x_min=50, y_min=150, x_max=150, y_max=190)
    assert vehicle.iou(plate) <= 0.1
    assert vehicle.containment_ratio(plate) == pytest.approx(1.0)


def test_make_model_prediction_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        MakeModelPrediction(ranked_labels=["a", "b"], ranked_confidences=[0.9])


def test_make_model_prediction_top_1() -> None:
    prediction = MakeModelPrediction(
        ranked_labels=["renault_clio", "fiat_egea"], ranked_confidences=[0.8, 0.2]
    )
    assert prediction.top_1 == "renault_clio"
