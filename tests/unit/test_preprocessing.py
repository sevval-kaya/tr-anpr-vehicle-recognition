import numpy as np

from plaka.ocr.preprocessing import enhance_plate_crop


def test_enhance_plate_crop_preserves_shape_and_dtype() -> None:
    crop = np.random.randint(0, 256, size=(40, 160, 3), dtype=np.uint8)
    enhanced = enhance_plate_crop(crop)
    assert enhanced.shape == crop.shape
    assert enhanced.dtype == crop.dtype


def test_enhance_plate_crop_increases_low_contrast_variance() -> None:
    # A flat, low-contrast crop should end up with more spread after CLAHE.
    crop = np.full((40, 160, 3), 128, dtype=np.uint8)
    crop[10:30, 40:120] = 135  # faint embossed-character-like region
    enhanced = enhance_plate_crop(crop)
    assert enhanced.astype(np.float32).std() >= crop.astype(np.float32).std()
