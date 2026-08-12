"""Plate-crop preprocessing before OCR (pipeline stage 4: "görüntü ön işleme").

Baseline scope note (see docs/decisions.md): true perspective correction
needs either an oriented bounding box or 4-corner keypoints for the plate,
neither of which an axis-aligned YOLO box provides. The baseline therefore
only does contrast enhancement; perspective correction is deferred to the
"zorlu senaryo" (angled-shot) improvement pass in roadmap stage 5, once it's
clear axis-aligned crops aren't enough.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def enhance_plate_crop(
    crop_bgr: NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """Apply CLAHE contrast enhancement to a plate crop, in-place-safe.

    Operates in the L channel of LAB color space so color information is
    preserved while local contrast (helpful for low-light / glare plates)
    is boosted.
    """
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    # cv2's stubs widen the dtype; cvtColor preserves uint8 for a uint8 input.
    return cast("NDArray[np.uint8]", cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR))
