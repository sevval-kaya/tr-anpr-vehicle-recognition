"""Shared video-source helpers for anything that walks frames out of a
cv2.VideoCapture — scripts/run_inference_video.py and the web app's
JobManager (src/plaka/web/jobs.py) both need the exact same rotation and
time-based sampling logic, so it lives here once instead of twice.

Rotation is opt-in and explicit, not auto-detected: cv2's
CAP_PROP_ORIENTATION_META/CAP_PROP_ORIENTATION_AUTO was tried against this
project's own two test clips and turned out unreliable — arac2.mp4 (already
correctly oriented) reports a 180° metadata tag, and arac3.mp4 (genuinely
stored rotated) reports none at all, so trusting metadata would silently
corrupt one clip or leave the other broken depending on which way it's
wrong (see docs/decisions.md). A caller-supplied degrees value is the only
option that can't guess wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

# 90/270 swap width and height; 180 doesn't. Callers that size a
# VideoWriter off the source capture's width/height need to know this.
_ROTATE_CODES: dict[int, int] = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
VALID_ROTATIONS = frozenset({0, 90, 180, 270})


def apply_rotation(frame: NDArray[np.uint8], rotate_degrees: int) -> NDArray[np.uint8]:
    """Rotate `frame` clockwise by `rotate_degrees` (0/90/180/270 only).
    0 returns the frame unchanged (no copy).
    """
    code = _ROTATE_CODES.get(rotate_degrees)
    if code is None:
        if rotate_degrees != 0:
            raise ValueError(f"rotate_degrees must be one of {sorted(VALID_ROTATIONS)}, got {rotate_degrees}")
        return frame
    # cv2's stubs type rotate()'s return as a looser ndarray (doesn't
    # preserve the uint8 dtype through the C++ binding) — cast back to
    # what it actually returns at runtime.
    return cast(NDArray[np.uint8], cv2.rotate(frame, code))


def rotates_dimensions(rotate_degrees: int) -> bool:
    """True if `rotate_degrees` swaps width and height (90 or 270)."""
    return rotate_degrees in (90, 270)


def resolve_frame_stride(
    fps: float, sample_interval_seconds: float | None, frame_stride: int
) -> int:
    """Turn a "process every N seconds" request into the frame_stride the
    existing "process every Nth frame" loop already knows how to use —
    frame_stride = round(fps * seconds), floored at 1. Falls back to
    `frame_stride` unchanged when `sample_interval_seconds` is None (the
    two options are mutually exclusive; sample_interval_seconds wins when
    both are given, since a caller only sets both by explicit override).
    """
    if sample_interval_seconds is None:
        return frame_stride
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be > 0")
    return max(1, round(fps * sample_interval_seconds))


@dataclass(frozen=True, slots=True)
class FrameSamplingPlan:
    """Bundles the two independent "which frames do we actually run the
    model on" knobs (rotation, time-based sampling) so both call sites
    apply them identically instead of re-deriving the same arithmetic.
    """

    frame_stride: int
    rotate_degrees: int = 0

    @classmethod
    def build(
        cls,
        fps: float,
        frame_stride: int = 1,
        sample_interval_seconds: float | None = None,
        rotate_degrees: int = 0,
    ) -> FrameSamplingPlan:
        return cls(
            frame_stride=resolve_frame_stride(fps, sample_interval_seconds, frame_stride),
            rotate_degrees=rotate_degrees,
        )

    def should_process(self, frame_index: int) -> bool:
        return frame_index % self.frame_stride == 0

    def prepare(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return apply_rotation(frame, self.rotate_degrees)


def timestamp_seconds(frame_index: int, fps: float) -> float:
    return frame_index / fps if fps > 0 else 0.0
