"""Semi-automated OCR label bootstrapping via the Claude API (vision).

Used to transcribe plate text for `data/external/user_plates/`, a
1,955-image dataset that has bbox labels but no ground-truth plate text
(see docs/decisions.md). This is a one-time *data preparation* tool run
from scripts/label_plates_with_claude.py — it is not part of the runtime
inference pipeline, which keeps using `plaka.ocr.plate_ocr.PlateOcr`
(PaddleOCR) for actual inference. Callers run the returned raw text through
`TurkishPlateValidator` to decide whether to keep it as a label.
"""

from __future__ import annotations

import base64
import re

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import anthropic
except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
    raise ImportError(
        "anthropic is required for claude_labeler; install with `pip install -e '.[labeling]'`"
    ) from exc

MODEL = "claude-opus-5"

UNREADABLE_MARKER = "OKUNAMIYOR"

_SYSTEM_PROMPT = (
    "Sana kırpılmış bir Türk araç plakası görüntüsü verilecek. Görevin SADECE "
    "plakada yazan karakterleri okumak.\n"
    "Kurallar:\n"
    "- Yanıtın SADECE plaka metni olsun, başka hiçbir açıklama ekleme.\n"
    "- Karakterleri soldan sağa, büyük harfle, plakada göründükleri gibi yaz "
    "(örn. '34 ABC 123').\n"
    "- Emin değilsen bile en olası okumayı ver.\n"
    f"- Görüntüde okunabilir bir plaka YOKSA (bulanık, kırpılmamış, boş vb.) "
    f"sadece {UNREADABLE_MARKER} yaz."
)

# Defensive cleanup for the disabled-thinking failure mode where a stray
# XML-ish tag can leak into the visible response (see claude-api skill,
# "Two failure modes when thinking is disabled").
_TAG_PATTERN = re.compile(r"<[^>]+>")


def _encode_jpeg_base64(image_bgr: NDArray[np.uint8]) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("failed to encode crop as JPEG")
    return base64.standard_b64encode(buffer.tobytes()).decode("utf-8")


def transcribe_plate_crop(
    client: anthropic.Anthropic,
    plate_crop_bgr: NDArray[np.uint8],
) -> str:
    """Ask Claude to transcribe a single plate crop.

    Returns the raw text Claude produced (including the UNREADABLE_MARKER
    sentinel, unfiltered) — the caller decides what to do with it.
    """
    image_b64 = _encode_jpeg_base64(plate_crop_bgr)

    response = client.messages.create(
        model=MODEL,
        max_tokens=40,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": "Bu plakayi oku."},
                ],
            }
        ],
    )

    if response.stop_reason == "refusal":
        return UNREADABLE_MARKER

    text = "".join(block.text for block in response.content if block.type == "text")
    return _TAG_PATTERN.sub("", text).strip()
