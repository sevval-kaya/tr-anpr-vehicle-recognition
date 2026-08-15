import numpy as np
import pytest

from plaka.pipeline.video_io import (
    FrameSamplingPlan,
    apply_rotation,
    resolve_frame_stride,
    rotates_dimensions,
    timestamp_seconds,
)


class TestApplyRotation:
    def test_zero_degrees_returns_frame_unchanged(self) -> None:
        frame = np.arange(24, dtype=np.uint8).reshape(4, 6, 1)
        result = apply_rotation(frame, 0)
        assert result is frame

    def test_90_degrees_swaps_width_and_height(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)  # height=10, width=20
        result = apply_rotation(frame, 90)
        assert result.shape == (20, 10, 3)

    def test_270_degrees_swaps_width_and_height(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        result = apply_rotation(frame, 270)
        assert result.shape == (20, 10, 3)

    def test_180_degrees_keeps_shape(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        result = apply_rotation(frame, 180)
        assert result.shape == (10, 20, 3)

    def test_90_and_270_are_opposite_directions(self) -> None:
        frame = np.arange(20, dtype=np.uint8).reshape(4, 5, 1)
        rotated_cw = apply_rotation(frame, 90)
        rotated_ccw = apply_rotation(frame, 270)
        assert not np.array_equal(rotated_cw, rotated_ccw)

    def test_invalid_degrees_raises(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_rotation(frame, 45)


class TestRotatesDimensions:
    def test_90_and_270_swap(self) -> None:
        assert rotates_dimensions(90) is True
        assert rotates_dimensions(270) is True

    def test_0_and_180_do_not_swap(self) -> None:
        assert rotates_dimensions(0) is False
        assert rotates_dimensions(180) is False


class TestResolveFrameStride:
    def test_none_interval_returns_frame_stride_unchanged(self) -> None:
        assert resolve_frame_stride(fps=30.0, sample_interval_seconds=None, frame_stride=5) == 5

    def test_one_second_interval_at_30fps(self) -> None:
        assert resolve_frame_stride(fps=30.0, sample_interval_seconds=1.0, frame_stride=1) == 30

    def test_one_second_interval_at_60fps(self) -> None:
        # This is the whole point of the feature: "1 second" means the same
        # wall-clock interval regardless of the source fps.
        assert resolve_frame_stride(fps=60.0, sample_interval_seconds=1.0, frame_stride=1) == 60

    def test_half_second_interval_rounds(self) -> None:
        # round() is banker's rounding: round(12.5) == 12, not 13.
        assert resolve_frame_stride(fps=25.0, sample_interval_seconds=0.5, frame_stride=1) == 12

    def test_floors_at_1(self) -> None:
        assert resolve_frame_stride(fps=1.0, sample_interval_seconds=0.01, frame_stride=1) == 1

    def test_zero_or_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_frame_stride(fps=30.0, sample_interval_seconds=0.0, frame_stride=1)


class TestFrameSamplingPlan:
    def test_build_prefers_sample_interval_over_frame_stride(self) -> None:
        plan = FrameSamplingPlan.build(fps=30.0, frame_stride=1, sample_interval_seconds=2.0)
        assert plan.frame_stride == 60

    def test_should_process_respects_stride(self) -> None:
        plan = FrameSamplingPlan(frame_stride=10)
        assert plan.should_process(0) is True
        assert plan.should_process(9) is False
        assert plan.should_process(10) is True

    def test_prepare_applies_rotation(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        plan = FrameSamplingPlan(frame_stride=1, rotate_degrees=90)
        assert plan.prepare(frame).shape == (20, 10, 3)


class TestTimestampSeconds:
    def test_basic(self) -> None:
        assert timestamp_seconds(30, fps=30.0) == 1.0

    def test_zero_fps_does_not_divide_by_zero(self) -> None:
        assert timestamp_seconds(30, fps=0.0) == 0.0
