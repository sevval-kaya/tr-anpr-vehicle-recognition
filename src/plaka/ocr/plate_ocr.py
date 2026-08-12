"""Plate character recognition (OCR) stage.

Wraps PaddleOCR, restricted to the Turkish plate charset (uppercase Latin
letters + digits) as recommended in docs/architecture.md section 3.2:
re-training/restricting the recognizer to this charset measurably improves
accuracy versus general-purpose OCR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

_NON_PLATE_CHARS = re.compile(r"[^A-Z0-9]")


@dataclass(frozen=True, slots=True)
class OcrReading:
    raw_text: str
    confidence: float


class PlateOcr:
    """Reads characters from a preprocessed plate crop.

    `weights_path=None` uses PaddleOCR's bundled general-purpose recognition
    weights (fine for early pipeline smoke-testing); pass a path to a
    Turkish-plate-fine-tuned recognizer once one has been trained
    (roadmap stage 4).
    """

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._model: PaddleOCR | None = None

    def _ensure_model_loaded(self) -> PaddleOCR:
        if self._model is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "paddleocr is required for PlateOcr; install with "
                    "`pip install -e '.[ocr]'`"
                ) from exc
            kwargs = {"rec_model_dir": str(self._weights_path)} if self._weights_path else {}
            self._model = PaddleOCR(lang="en", use_angle_cls=False, **kwargs)
        return self._model

    def read(self, plate_crop_bgr: NDArray[np.uint8]) -> OcrReading:
        """Run OCR on a single preprocessed plate crop, returning the best reading.

        Returns an empty-text, zero-confidence reading if nothing was
        recognized, rather than raising — callers (the format validator)
        treat that as a rejected read like any other.
        """
        model = self._ensure_model_loaded()
        result = model.ocr(plate_crop_bgr, cls=False)

        if not result or not result[0]:
            return OcrReading(raw_text="", confidence=0.0)

        # Multiple text fragments can be detected on one crop (e.g. il kodu
        # separated from the rest); concatenate in left-to-right reading
        # order and average confidence.
        fragments = sorted(result[0], key=lambda item: item[0][0][0])
        texts = [_NON_PLATE_CHARS.sub("", text.upper()) for _, (text, _confidence) in fragments]
        confidences = [confidence for _, (_text, confidence) in fragments]

        combined_text = "".join(texts)
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return OcrReading(raw_text=combined_text, confidence=mean_confidence)
