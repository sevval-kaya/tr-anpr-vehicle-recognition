"""Shared data contracts passed between pipeline stages.

These types are the interface boundary between detection, OCR, and
classification modules, so each module can be developed/tested/swapped
independently (see docs/architecture.md, section "Uçtan uca işlem hattı").
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned pixel box in the coordinate space of the source frame."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def _check_ordering(self) -> BoundingBox:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                f"invalid box: x_max/y_max must exceed x_min/y_min, got {self!r}"
            )
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def _intersection_area(self, other: BoundingBox) -> float:
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)

        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)
        return inter_width * inter_height

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union with another box."""
        intersection = self._intersection_area(other)
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union if union > 0 else 0.0

    def containment_ratio(self, other: BoundingBox) -> float:
        """Fraction of `other`'s area that falls inside this box.

        Used to associate a small detected plate box with the (much
        larger) vehicle box it belongs to: IoU is a poor fit for that
        because the vehicle's area dominates the union, so a correctly
        contained plate can still score a near-zero IoU. Containment
        directly answers "how much of the plate sits inside this
        vehicle" regardless of the vehicle's size.
        """
        other_area = other.width * other.height
        if other_area <= 0:
            return 0.0
        return self._intersection_area(other) / other_area


class PlateReading(BaseModel):
    """Result of plate detection + OCR + format validation for one plate."""

    box: BoundingBox
    raw_text: str
    normalized_text: str | None = None
    is_format_valid: bool
    detection_confidence: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class MakeModelPrediction(BaseModel):
    """Top-k make/model classification result for one vehicle crop."""

    ranked_labels: list[str]
    ranked_confidences: list[float] = Field(
        description="Confidence per label in ranked_labels, same order, descending."
    )

    @model_validator(mode="after")
    def _check_same_length(self) -> MakeModelPrediction:
        if len(self.ranked_labels) != len(self.ranked_confidences):
            raise ValueError("ranked_labels and ranked_confidences must be the same length")
        return self

    @property
    def top_1(self) -> str | None:
        return self.ranked_labels[0] if self.ranked_labels else None


class VehicleDetection(BaseModel):
    """One detected vehicle within a frame, with its associated plate (if any)
    and make/model prediction (if classification was run)."""

    box: BoundingBox
    vehicle_type: str = Field(
        description="COCO vehicle category from VehicleDetector, e.g. 'car'/'motorcycle'/'bus'/'truck'."
    )
    detection_confidence: float = Field(ge=0.0, le=1.0)
    plate: PlateReading | None = None
    make_model: MakeModelPrediction | None = Field(
        default=None,
        description="Set only when classification is enabled (configs/pipeline.yaml "
        "classification.enabled) — see docs/decisions.md #29. Kept in the schema so "
        "the make/model path can be re-enabled later without a data-model change.",
    )
    track_id: int | None = Field(
        default=None,
        description="Stable ID across frames, set once temporal voting is wired in.",
    )


class FrameResult(BaseModel):
    """Full pipeline output for a single frame."""

    frame_index: int
    vehicles: list[VehicleDetection] = Field(default_factory=list)
