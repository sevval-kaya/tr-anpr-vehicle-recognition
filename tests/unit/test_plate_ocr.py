import numpy as np

from plaka.ocr.plate_ocr import OcrReading, _select_plate_text, _upscale_if_small


class TestUpscaleIfSmall:
    def test_short_crop_is_upscaled_to_min_height(self) -> None:
        image = np.zeros((100, 400, 3), dtype=np.uint8)
        result = _upscale_if_small(image, min_height=200)
        assert result.shape[0] == 200
        assert result.shape[1] == 800  # aspect ratio preserved (2x)

    def test_tall_enough_crop_is_returned_unchanged(self) -> None:
        image = np.zeros((250, 900, 3), dtype=np.uint8)
        result = _upscale_if_small(image, min_height=200)
        assert result.shape == image.shape

    def test_zero_height_image_is_returned_unchanged(self) -> None:
        image = np.zeros((0, 400, 3), dtype=np.uint8)
        result = _upscale_if_small(image, min_height=200)
        assert result.shape == image.shape


class TestSelectPlateText:
    def test_empty_input_returns_empty_reading(self) -> None:
        result = _select_plate_text([], [], [])
        assert result == OcrReading(raw_text="", confidence=0.0)

    def test_single_region_is_returned_as_is(self) -> None:
        result = _select_plate_text(["34 AB 123"], [0.9], [(0.0, 0.0, 100.0, 30.0)])
        assert result.raw_text == "34AB123"
        assert result.confidence == 0.9

    def test_country_badge_is_excluded_even_when_same_row(self) -> None:
        # Real case (s_477.jpg): "55", "DN", "079" are fragments of one
        # plate line; "TR" is the country-code badge sharing the same
        # vertical band and must not be merged in.
        texts = ["DN", "079", "55", "TR"]
        scores = [1.0, 1.0, 1.0, 1.0]
        boxes = [
            (170.0, 31.0, 253.0, 103.0),
            (268.0, 23.0, 381.0, 103.0),
            (70.0, 32.0, 151.0, 108.0),
            (49.0, 72.0, 81.0, 100.0),
        ]
        result = _select_plate_text(texts, scores, boxes)
        assert result.raw_text == "55DN079"

    def test_plate_line_split_across_two_boxes_is_merged(self) -> None:
        # Real case (s_253.jpg): il kodu detected as its own box, separate
        # from the harf+rakam group.
        texts = ["38", "PD369", "TR"]
        scores = [1.0, 1.0, 1.0]
        boxes = [
            (196.0, 114.0, 404.0, 261.0),
            (411.0, 100.0, 959.0, 269.0),
            (145.0, 191.0, 210.0, 242.0),
        ]
        result = _select_plate_text(texts, scores, boxes)
        assert result.raw_text == "38PD369"

    def test_dealer_and_city_footer_text_is_excluded(self) -> None:
        # Real case (s_197.jpg): footer rows sit below the plate line and
        # must not be merged in, even though the "TR" badge and plate line
        # do overlap vertically.
        texts = ["SH 050", "TR", "KAYSERI", "OTOBANK", "0352 240 12 53"]
        scores = [1.0, 1.0, 0.996, 1.0, 0.998]
        boxes = [
            (324.0, 68.0, 712.0, 209.0),
            (109.0, 154.0, 155.0, 193.0),
            (107.0, 214.0, 192.0, 239.0),
            (309.0, 207.0, 493.0, 240.0),
            (575.0, 205.0, 713.0, 235.0),
        ]
        result = _select_plate_text(texts, scores, boxes)
        assert result.raw_text == "SH050"

    def test_smaller_font_footer_text_is_excluded_despite_same_row(self) -> None:
        # Real case (25_0.jpg): the plate line's box is tall enough that
        # small-font dealer/city footer text below it still has a
        # y-center inside the anchor's y-range — must be excluded by the
        # font-size check, not just row position.
        texts = ["06 DN 1026", "TR", "Audi", "KARACA", "0354", "22 82 81 YOZGAT", "0542"]
        scores = [0.974, 1.0, 0.994, 1.0, 1.0, 0.847, 0.989]
        boxes = [
            (140.0, 63.0, 713.0, 205.0),
            (108.0, 144.0, 157.0, 182.0),
            (142.0, 198.0, 226.0, 228.0),
            (243.0, 190.0, 484.0, 227.0),
            (504.0, 193.0, 550.0, 207.0),
            (538.0, 185.0, 704.0, 221.0),
            (502.0, 200.0, 548.0, 221.0),
        ]
        result = _select_plate_text(texts, scores, boxes)
        assert result.raw_text == "06DN1026"

    def test_low_confidence_noise_fragment_is_excluded(self) -> None:
        # Real case (s_533.jpg): a small circular emblem next to the "TR"
        # badge was misread as a stray "E" (score 0.564) whose height
        # happens to be within the font-size ratio of the anchor.
        texts = ["E", "34 ABD 987"]
        scores = [0.564, 0.990]
        boxes = [(39.0, 23.0, 58.0, 57.0), (49.0, 13.0, 266.0, 63.0)]
        result = _select_plate_text(texts, scores, boxes)
        assert result.raw_text == "34ABD987"

    def test_country_badge_glued_onto_a_single_merged_region_is_stripped(self) -> None:
        # Real case (860.jpg with crop padding added): the detector
        # returned "TR66LN948" as one region rather than "TR" separate
        # from "66 LN 948" — the per-fragment exclusion can't catch this,
        # so the merged text itself must have the prefix stripped.
        result = _select_plate_text(["TR66LN948"], [0.97], [(0.0, 0.0, 400.0, 80.0)])
        assert result.raw_text == "66LN948"

    def test_missing_boxes_falls_back_to_first_nonempty_region(self) -> None:
        # No box data at all (e.g. an unexpected result shape): every
        # region is treated as its own anchor with zero area, so the loop
        # still returns *something* rather than crashing.
        result = _select_plate_text(["34 AB 123"], [0.5], [])
        assert result.raw_text == "34AB123"
