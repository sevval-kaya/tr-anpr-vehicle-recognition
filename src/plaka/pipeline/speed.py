"""Uncalibrated, self-referenced-scale speed estimation for tracked vehicles.

No camera calibration (focal length, known reference distance, homography)
is available or planned for this project — see docs/decisions.md #42. A
single global "pixels = X meters" constant would be even less reliable
(mixes near/far perspective error into one number), so instead each
vehicle's *own* detected box width is used as a running reference for how
many meters a pixel spans at that vehicle's current distance from the
camera. This is still far from a true speed measurement (see decisions.md
#42 for the full list of error sources) — every value derived here is
explicitly presented to the user as an approximation, never a precise
reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Rough population-average vehicle lengths in meters, used only as the
# per-vehicle-type reference for the self-referenced pixel scale below.
# Not measured for this project — commonly cited approximate figures for
# each COCO vehicle category (see docs/decisions.md #42 for sourcing notes
# and why these are treated as rough constants, not calibration data).
ASSUMED_VEHICLE_LENGTH_M: dict[str, float] = {
    "car": 4.5,
    "motorcycle": 2.0,
    "bus": 12.0,
    "truck": 8.0,
}

# Speeds outside this range are treated as detector/tracker jitter, not a
# real reading, and excluded from the moving average entirely (see module
# docstring and docs/decisions.md #42) — a small/angled crop or a one-frame
# box jump can imply a wildly wrong instantaneous speed.
MIN_PLAUSIBLE_KMH = 0.0
MAX_PLAUSIBLE_KMH = 200.0

# How many of the most recent position observations feed the moving
# average (i.e. up to this many consecutive pairs) — smooths per-frame
# jitter while still reacting to a real speed change within a few seconds.
SPEED_WINDOW_OBSERVATIONS = 10

_METERS_PER_SECOND_TO_KMH = 3.6


@dataclass(frozen=True, slots=True)
class BoxPositionObservation:
    """One frame's position sample for a tracked vehicle: the box's
    bottom-center point (an approximate ground-contact point) plus its
    pixel width (the self-referenced distance scale for that frame).
    """

    timestamp_seconds: float
    bottom_center_x: float
    bottom_center_y: float
    box_width_px: float


def _pairwise_speed_kmh(
    a: BoxPositionObservation, b: BoxPositionObservation, assumed_length_m: float
) -> float | None:
    """Speed implied by two consecutive observations, or None if the pair
    can't yield a trustworthy value (non-positive time/width gap, or an
    implausible result — see MIN/MAX_PLAUSIBLE_KMH).
    """
    delta_t = b.timestamp_seconds - a.timestamp_seconds
    if delta_t <= 0:
        return None

    avg_width_px = (a.box_width_px + b.box_width_px) / 2
    if avg_width_px <= 0:
        return None

    meters_per_pixel = assumed_length_m / avg_width_px
    dx = b.bottom_center_x - a.bottom_center_x
    dy = b.bottom_center_y - a.bottom_center_y
    pixel_distance = (dx**2 + dy**2) ** 0.5
    speed_kmh = (pixel_distance * meters_per_pixel / delta_t) * _METERS_PER_SECOND_TO_KMH

    if not (MIN_PLAUSIBLE_KMH <= speed_kmh <= MAX_PLAUSIBLE_KMH):
        return None
    return float(speed_kmh)


def estimate_speed_kmh(
    observations: Sequence[BoxPositionObservation], vehicle_type: str
) -> float | None:
    """Moving-average speed estimate over the last SPEED_WINDOW_OBSERVATIONS
    consecutive pairs of `observations` (oldest-to-newest order).

    Returns None until at least two observations exist, or if every
    candidate pair was filtered out as implausible (see
    _pairwise_speed_kmh) — callers should treat None as "no estimate yet",
    not "zero".
    """
    if len(observations) < 2:
        return None

    assumed_length_m = ASSUMED_VEHICLE_LENGTH_M.get(
        vehicle_type, ASSUMED_VEHICLE_LENGTH_M["car"]
    )
    windowed = list(observations)[-(SPEED_WINDOW_OBSERVATIONS + 1) :]

    speeds = [
        speed
        # strict=False: this is the standard "consecutive pairs" idiom
        # (windowed[1:] is deliberately one element shorter), not a bug.
        for a, b in zip(windowed, windowed[1:], strict=False)
        if (speed := _pairwise_speed_kmh(a, b, assumed_length_m)) is not None
    ]
    if not speeds:
        return None
    return sum(speeds) / len(speeds)


def relative_speed_label(speed_kmh: float) -> str:
    """Coarse, deliberately imprecise "yavaş / normal / hızlı" label —
    meant to counterbalance the false precision a bare km/h number implies
    (see docs/decisions.md #42). Thresholds are arbitrary round numbers,
    not derived from any measurement.
    """
    if speed_kmh < 15:
        return "yavaş"
    if speed_kmh <= 60:
        return "normal"
    return "hızlı"


def exceeds_speed_limit(speed_kmh: float | None, speed_limit_kmh: float) -> bool:
    return speed_kmh is not None and speed_kmh > speed_limit_kmh
