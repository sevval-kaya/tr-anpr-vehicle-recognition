import numpy as np
import pytest

from plaka.data.plate_crop import (
    crop_plate_from_yolo_box,
    largest_box_line,
    parse_yolo_label_line,
)


class TestParseYoloLabelLine:
    def test_parses_five_fields(self) -> None:
        class_id, x, y, w, h = parse_yolo_label_line("0 0.5 0.4 0.2 0.1")
        assert (class_id, x, y, w, h) == (0, 0.5, 0.4, 0.2, 0.1)

    def test_wrong_field_count_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_yolo_label_line("0 0.5 0.4")


class TestLargestBoxLine:
    def test_picks_the_larger_area_box(self) -> None:
        text = "0 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.4 0.4\n"
        assert largest_box_line(text) == "0 0.3 0.3 0.4 0.4"

    def test_skips_blank_lines(self) -> None:
        text = "\n0 0.5 0.5 0.2 0.2\n\n"
        assert largest_box_line(text) == "0 0.5 0.5 0.2 0.2"

    def test_empty_text_returns_none(self) -> None:
        assert largest_box_line("") is None
        assert largest_box_line("   \n  ") is None


class TestCropPlateFromYoloBox:
    def _image(self, width: int = 100, height: int = 50) -> np.ndarray:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Distinct value per pixel makes it easy to assert exact crop bounds.
        image[:, :] = np.arange(width).reshape(1, width, 1) % 256
        return image

    def test_crops_expected_region_with_padding(self) -> None:
        image = self._image(width=100, height=50)
        # Box centered at (50, 25), 20x10 in pixels -> normalized 0.2 x 0.2.
        crop = crop_plate_from_yolo_box(
            image, x_center=0.5, y_center=0.5, width=0.2, height=0.2, padding_ratio=0.0
        )
        assert crop.shape[:2] == (10, 20)

    def test_padding_expands_the_crop(self) -> None:
        image = self._image(width=100, height=50)
        no_pad = crop_plate_from_yolo_box(
            image, x_center=0.5, y_center=0.5, width=0.2, height=0.2, padding_ratio=0.0
        )
        padded = crop_plate_from_yolo_box(
            image, x_center=0.5, y_center=0.5, width=0.2, height=0.2, padding_ratio=0.5
        )
        assert padded.shape[0] > no_pad.shape[0]
        assert padded.shape[1] > no_pad.shape[1]

    def test_box_at_edge_is_clamped_not_negative(self) -> None:
        image = self._image(width=100, height=50)
        crop = crop_plate_from_yolo_box(
            image, x_center=0.0, y_center=0.0, width=0.2, height=0.4, padding_ratio=0.2
        )
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0
        assert crop.shape[0] <= image.shape[0]
        assert crop.shape[1] <= image.shape[1]
