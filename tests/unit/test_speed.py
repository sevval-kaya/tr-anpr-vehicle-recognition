import pytest

from plaka.pipeline.speed import (
    ASSUMED_VEHICLE_LENGTH_M,
    MAX_PLAUSIBLE_KMH,
    BoxPositionObservation,
    estimate_speed_kmh,
    exceeds_speed_limit,
    relative_speed_label,
)


def _observation(
    t: float, x: float, width: float = 100.0, y: float = 200.0
) -> BoxPositionObservation:
    return BoxPositionObservation(
        timestamp_seconds=t, bottom_center_x=x, bottom_center_y=y, box_width_px=width
    )


class TestEstimateSpeedKmh:
    def test_fewer_than_two_observations_yields_none(self) -> None:
        assert estimate_speed_kmh([], "car") is None
        assert estimate_speed_kmh([_observation(0.0, 0.0)], "car") is None

    def test_two_observations_yield_a_plausible_speed(self) -> None:
        # Box width 100px stays constant -> meters_per_pixel = 4.5/100 = 0.045.
        # 50px displacement in 1s -> 2.25 m/s -> 8.1 km/h.
        observations = [_observation(0.0, 0.0), _observation(1.0, 50.0)]
        speed = estimate_speed_kmh(observations, "car")
        assert speed is not None
        assert speed == pytest.approx(8.1)

    def test_stationary_vehicle_yields_zero(self) -> None:
        observations = [_observation(0.0, 100.0), _observation(1.0, 100.0)]
        assert estimate_speed_kmh(observations, "car") == 0.0

    def test_moving_average_smooths_a_single_outlier_frame(self) -> None:
        # Steady ~10px/frame motion at 10fps except one huge jitter jump
        # (simulates a tracker/detector box snapping briefly) — the jump's
        # implied speed is filtered out (see MAX_PLAUSIBLE_KMH), so the
        # average stays close to the steady value instead of being skewed.
        observations = [_observation(i * 0.1, i * 10.0) for i in range(5)]
        observations.append(_observation(0.5 + 0.001, 5000.0))  # implausible jump
        observations.append(_observation(0.6 + 0.001, 5010.0))
        speed = estimate_speed_kmh(observations, "car")
        assert speed is not None
        assert 0 < speed < MAX_PLAUSIBLE_KMH
        # Should be close to the steady-motion speed, not dragged toward
        # whatever a >200km/h outlier pair would otherwise average in.
        steady_speed = estimate_speed_kmh(observations[:5], "car")
        assert steady_speed is not None
        assert abs(speed - steady_speed) < 5.0

    def test_never_produces_absurd_values(self) -> None:
        # A single implausible jump between the only two observations has
        # nothing else to average with -> no trustworthy estimate at all.
        observations = [_observation(0.0, 0.0), _observation(0.001, 10000.0)]
        assert estimate_speed_kmh(observations, "car") is None

    def test_zero_or_negative_time_gap_is_ignored(self) -> None:
        observations = [_observation(1.0, 0.0), _observation(1.0, 500.0)]
        assert estimate_speed_kmh(observations, "car") is None

    def test_unknown_vehicle_type_falls_back_to_car_length(self) -> None:
        observations = [_observation(0.0, 0.0), _observation(1.0, 50.0)]
        assert estimate_speed_kmh(observations, "unknown_type") == estimate_speed_kmh(
            observations, "car"
        )

    def test_all_four_vehicle_types_have_assumed_lengths(self) -> None:
        assert set(ASSUMED_VEHICLE_LENGTH_M) == {"car", "motorcycle", "bus", "truck"}


class TestRelativeSpeedLabel:
    def test_thresholds(self) -> None:
        assert relative_speed_label(5) == "yavaş"
        assert relative_speed_label(30) == "normal"
        assert relative_speed_label(90) == "hızlı"


class TestExceedsSpeedLimit:
    def test_none_speed_never_exceeds(self) -> None:
        assert exceeds_speed_limit(None, 50.0) is False

    def test_above_and_below_limit(self) -> None:
        assert exceeds_speed_limit(60.0, 50.0) is True
        assert exceeds_speed_limit(40.0, 50.0) is False
        assert exceeds_speed_limit(50.0, 50.0) is False  # exactly at limit -> not exceeded
