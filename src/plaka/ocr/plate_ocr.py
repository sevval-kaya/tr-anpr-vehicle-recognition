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

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

_NON_PLATE_CHARS = re.compile(r"[^A-Z0-9]")

# The EU/TR country-code badge is a fixed, non-plate-text fixture that the
# detector frequently isolates as its own region, positioned inside the
# same vertical band as the actual plate line — so it can't be filtered
# out by row-position alone (see _select_plate_text) and is instead
# excluded by literal text match.
_COUNTRY_BADGE_TEXT = "TR"

# A same-line fragment of the plate text (e.g. the il kodu split from the
# rest) is set in the same font/height as the anchor region; dealer/city
# frame branding sits in a visibly smaller font, even when its y-center
# falls inside the anchor's y-range on a short/tightly-cropped plate.
_MIN_FRAGMENT_HEIGHT_RATIO = 0.5

# Recognizer confidence below this is treated as noise (e.g. a logo/emblem
# misread as a stray character) rather than a real, if uncertain, fragment.
_MIN_FRAGMENT_CONFIDENCE = 0.6

# PaddleOCR's text detector can miss an otherwise perfectly legible plate
# line entirely (zero regions returned, not a bad read) when the crop is
# too small in absolute pixels — confirmed empirically: the same plate
# went from 0 detected regions at 113px crop height to a clean read at
# 170px+, regardless of how much border/margin surrounds it (see
# docs/decisions.md). Crops shorter than this are upscaled before OCR.
_MIN_CROP_HEIGHT_PX = 200


def _upscale_if_small(
    image: NDArray[np.uint8], min_height: int = _MIN_CROP_HEIGHT_PX
) -> NDArray[np.uint8]:
    """Upscale `image` (preserving aspect ratio) if it's shorter than
    `min_height`; returns it unchanged otherwise.
    """
    height = image.shape[0]
    if height <= 0 or height >= min_height:
        return image
    scale = min_height / height
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


@dataclass(frozen=True, slots=True)
class OcrReading:
    raw_text: str
    confidence: float


def _select_plate_text(
    texts: list[str],
    scores: list[float],
    boxes: list[tuple[float, float, float, float]],
) -> OcrReading:
    """Pick and merge the detected text region(s) that make up the plate
    number line, out of every region PaddleOCR's detector found in a plate
    crop (which can include the plate line split across multiple boxes,
    plus non-plate fixtures: the country-code badge, dealer/city frame
    branding, phone numbers).

    Strategy: the plate line's largest single detected region anchors the
    read. Another region is merged into it (in left-to-right/x order) only
    if it clears three checks — same row (y-center falls inside the
    anchor's y-range), similar font size (height at least half the
    anchor's — dealer/city frame branding is set in a visibly smaller
    font, and its y-center can still land inside a tall anchor's y-range
    on a tightly-cropped plate), and confident (score above a noise
    floor, e.g. rejecting a logo/emblem misread as a stray character) —
    or a literal "TR" region, which is the country-code badge and can
    pass every check above (same height and confidence as real plate
    text) so is instead excluded by literal text match.
    """
    if not texts:
        return OcrReading(raw_text="", confidence=0.0)
    if len(boxes) != len(texts):
        boxes = [(0.0, 0.0, 0.0, 0.0)] * len(texts)

    areas = [(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max in boxes]
    anchor_index = max(range(len(texts)), key=lambda i: areas[i])
    _anchor_x_min, anchor_y_min, _anchor_x_max, anchor_y_max = boxes[anchor_index]
    anchor_height = anchor_y_max - anchor_y_min

    fragments: list[tuple[float, str, float]] = []  # (x_min, text, score)
    for index, (text, score, box) in enumerate(zip(texts, scores, boxes, strict=True)):
        cleaned = _NON_PLATE_CHARS.sub("", text.upper())
        if not cleaned or cleaned == _COUNTRY_BADGE_TEXT:
            continue
        x_min, y_min, _x_max, y_max = box
        is_anchor = index == anchor_index
        if is_anchor:
            fragments.append((x_min, cleaned, score))
            continue
        y_center = (y_min + y_max) / 2
        height = y_max - y_min
        same_line = anchor_y_min <= y_center <= anchor_y_max
        similar_font_size = anchor_height <= 0 or height >= _MIN_FRAGMENT_HEIGHT_RATIO * anchor_height
        confident = score >= _MIN_FRAGMENT_CONFIDENCE
        if same_line and similar_font_size and confident:
            fragments.append((x_min, cleaned, score))

    if not fragments:
        return OcrReading(raw_text="", confidence=0.0)

    fragments.sort(key=lambda fragment: fragment[0])
    raw_text = "".join(text for _x, text, _score in fragments)
    mean_confidence = sum(score for _x, _text, score in fragments) / len(fragments)

    # The country-code badge can also come back glued onto the front of a
    # single merged region (e.g. "TR66LN948") rather than as its own
    # fragment, most often with extra crop padding — the per-fragment
    # exclusion above only catches it when it's separate. A Turkish plate
    # always starts with the (numeric) il kodu, so a literal "TR" prefix
    # is unambiguously the badge, never real plate content.
    if raw_text.startswith(_COUNTRY_BADGE_TEXT):
        raw_text = raw_text[len(_COUNTRY_BADGE_TEXT) :]

    return OcrReading(raw_text=raw_text, confidence=mean_confidence)


class PlateOcr:
    """Reads characters from a preprocessed plate crop.

    `weights_path=None` uses PaddleOCR's bundled general-purpose recognition
    weights (fine for early pipeline smoke-testing); pass a path to a
    Turkish-plate-fine-tuned recognizer directory once one has been trained
    (roadmap stage 4).

    Uses PaddleOCR's PP-OCRv6 pipeline (`predict()`), not the older
    `PaddleOCR.ocr()` API (removed in paddleocr 3.x — see
    docs/decisions.md). Document-orientation classification and unwarping
    are disabled: plate crops are already axis-aligned and small, so those
    stages only add latency. `enable_mkldnn=False` works around a CPU
    inference crash (`NotImplementedError` in oneDNN's PIR attribute
    conversion) observed with paddlepaddle 3.3.1 on this project's dev
    machine; drop it if a future paddlepaddle release fixes the crash and
    MKL-DNN's speedup is wanted back. Crops shorter than
    `_MIN_CROP_HEIGHT_PX` are upscaled before OCR — below that, the
    detector can miss the plate line entirely (zero regions), not just
    read it poorly.

    Uses the PP-OCRv6 "tiny" det/rec pair, not the "medium" pair PaddleOCR
    defaults to — this machine has no GPU-enabled paddlepaddle build (CPU
    only), and "medium" measured ~2.2-3.2s per OCR call on CPU here, which
    made the live-camera web view unusably laggy (a whole frame round trip
    could take 1.5s+ with two plates in view). Benchmarked against the same
    42+10 human-verified real plate crops used for the original OCR
    baseline (docs/decisions.md #21): "tiny" is ~13x faster (~200ms/call)
    and *matches or beats* "medium" on the easy set (100% exact vs 97.6%,
    0% CER vs 0.64%); on the 10-image hard/zorlu set it's very slightly
    behind (90% exact vs 100%, one additional miss) — an acceptable trade
    for making live camera/video actually usable (see docs/decisions.md).
    """

    _DEFAULT_DET_MODEL = "PP-OCRv6_tiny_det"
    _DEFAULT_REC_MODEL = "PP-OCRv6_tiny_rec"

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
            weights_kwargs = (
                {"text_recognition_model_dir": str(self._weights_path)}
                if self._weights_path
                else {}
            )
            self._model = PaddleOCR(
                # `lang` is ignored once explicit model names are given
                # (PaddleOCR warns on this) — text_recognition_model_name
                # already pins the English-charset "tiny" recognizer below.
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                text_detection_model_name=self._DEFAULT_DET_MODEL,
                text_recognition_model_name=self._DEFAULT_REC_MODEL,
                **weights_kwargs,
            )
        return self._model

    def read(self, plate_crop_bgr: NDArray[np.uint8]) -> OcrReading:
        """Run OCR on a single preprocessed plate crop, returning the best reading.

        Returns an empty-text, zero-confidence reading if nothing was
        recognized, rather than raising — callers (the format validator)
        treat that as a rejected read like any other.
        """
        model = self._ensure_model_loaded()
        results = model.predict(_upscale_if_small(plate_crop_bgr))

        if not results:
            return OcrReading(raw_text="", confidence=0.0)

        page = results[0]
        texts: list[str] = page.get("rec_texts") or []
        scores: list[float] = page.get("rec_scores") or []
        raw_boxes = page.get("rec_boxes")
        boxes = (
            [tuple(float(v) for v in box) for box in raw_boxes]
            if raw_boxes is not None
            else []
        )
        return _select_plate_text(texts, scores, boxes)
