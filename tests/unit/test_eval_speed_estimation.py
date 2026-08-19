import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from plaka.pipeline.schemas import BoundingBox, FrameResult, VehicleDetection

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from eval_speed_estimation import ClipResult, evaluate, ground_truth_kmh  # noqa: E402


def _write_tiny_video(path: Path, frame_count: int, fps: float, size: tuple[int, int]) -> None:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()


@dataclass
class _MovingVehiclePipeline:
    """Same box every call, shifted right by `shift_px_per_call` each
    time, optionally with a short-lived second ("decoy") vehicle in the
    first few calls — for exercising dominant-track selection."""

    shift_px_per_call: float
    decoy_calls: int = 0
    box_width_px: float = 20.0
    calls: int = field(default=0, init=False)

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        x0 = self.calls * self.shift_px_per_call
        vehicles = [
            VehicleDetection(
                box=BoundingBox(x_min=x0, y_min=0, x_max=x0 + self.box_width_px, y_max=15),
                vehicle_type="car",
                detection_confidence=0.9,
                plate=None,
            )
        ]
        if self.calls < self.decoy_calls:
            vehicles.append(
                VehicleDetection(
                    box=BoundingBox(x_min=150, y_min=50, x_max=170, y_max=65),
                    vehicle_type="truck",
                    detection_confidence=0.8,
                    plate=None,
                )
            )
        self.calls += 1
        return FrameResult(frame_index=frame_index, vehicles=vehicles)


class TestGroundTruthKmh:
    def test_basic_formula(self) -> None:
        # 15m in 1.5s -> 10 m/s -> 36 km/h.
        assert ground_truth_kmh(15.0, 1.5) == pytest.approx(36.0)


class TestClipResultErrors:
    def test_abs_and_percent_error(self) -> None:
        result = ClipResult(
            video_path="a.mp4", ground_truth_kmh=40.0, predicted_kmh=44.0,
            vehicle_type="car", tracked_frames=10, other_tracks=0, notes="",
        )
        assert result.abs_error_kmh == pytest.approx(4.0)
        assert result.percent_error == pytest.approx(10.0)

    def test_none_predicted_yields_none_errors(self) -> None:
        result = ClipResult(
            video_path="a.mp4", ground_truth_kmh=40.0, predicted_kmh=None,
            vehicle_type=None, tracked_frames=0, other_tracks=0, notes="",
        )
        assert result.abs_error_kmh is None
        assert result.percent_error is None


class TestEvaluateEndToEnd:
    def _write_manifest(
        self, tmp_path: Path, rows: list[dict[str, str]], extra_columns: bool = True
    ) -> Path:
        manifest_path = tmp_path / "manifest.csv"
        fieldnames = ["video_path", "distance_m", "duration_seconds"]
        if extra_columns:
            fieldnames += ["rotate_degrees", "notes"]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return manifest_path

    def test_single_clip_produces_a_prediction_and_result_csv(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        _write_tiny_video(video_path, frame_count=10, fps=10.0, size=(200, 100))
        manifest = self._write_manifest(
            tmp_path,
            [{"video_path": "clip.mp4", "distance_m": "15.0", "duration_seconds": "1.5",
              "rotate_degrees": "0", "notes": "test"}],
        )
        output_csv = tmp_path / "results.csv"

        results = evaluate(
            manifest, output_csv, pipeline=_MovingVehiclePipeline(shift_px_per_call=5.0)
        )

        assert len(results) == 1
        r = results[0]
        assert r.ground_truth_kmh == pytest.approx(36.0)
        assert r.predicted_kmh is not None
        assert 0 < r.predicted_kmh < 200
        assert r.vehicle_type == "car"
        assert r.other_tracks == 0

        assert output_csv.exists()
        rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["video_path"] == "clip.mp4"
        assert rows[0]["notes"] == "test"

    def test_manifest_without_optional_columns_uses_defaults(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        _write_tiny_video(video_path, frame_count=6, fps=10.0, size=(200, 100))
        manifest = self._write_manifest(
            tmp_path,
            [{"video_path": "clip.mp4", "distance_m": "10.0", "duration_seconds": "2.0"}],
            extra_columns=False,
        )
        output_csv = tmp_path / "results.csv"

        results = evaluate(
            manifest, output_csv, pipeline=_MovingVehiclePipeline(shift_px_per_call=3.0)
        )

        assert len(results) == 1
        assert results[0].notes == ""

    def test_missing_clip_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [{"video_path": "does_not_exist.mp4", "distance_m": "10.0", "duration_seconds": "2.0",
              "rotate_degrees": "0", "notes": ""}],
        )
        output_csv = tmp_path / "results.csv"

        results = evaluate(
            manifest, output_csv, pipeline=_MovingVehiclePipeline(shift_px_per_call=3.0)
        )

        assert results == []

    def test_dominant_track_ignores_a_short_lived_decoy(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        _write_tiny_video(video_path, frame_count=10, fps=10.0, size=(200, 100))
        manifest = self._write_manifest(
            tmp_path,
            [{"video_path": "clip.mp4", "distance_m": "15.0", "duration_seconds": "1.5",
              "rotate_degrees": "0", "notes": ""}],
        )
        output_csv = tmp_path / "results.csv"

        # Decoy truck is present for only 2 frames (< 5), so it must not
        # trip the other_tracks warning even though it's a second vehicle.
        results = evaluate(
            manifest, output_csv,
            pipeline=_MovingVehiclePipeline(shift_px_per_call=5.0, decoy_calls=2),
        )

        assert results[0].vehicle_type == "car"
        assert results[0].other_tracks == 0

    def test_sustained_second_vehicle_is_flagged(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        _write_tiny_video(video_path, frame_count=10, fps=10.0, size=(200, 100))
        manifest = self._write_manifest(
            tmp_path,
            [{"video_path": "clip.mp4", "distance_m": "15.0", "duration_seconds": "1.5",
              "rotate_degrees": "0", "notes": ""}],
        )
        output_csv = tmp_path / "results.csv"

        # Decoy present for 8 of 10 frames -> long enough to plausibly be
        # confused with the real test subject; should be flagged.
        results = evaluate(
            manifest, output_csv,
            pipeline=_MovingVehiclePipeline(shift_px_per_call=5.0, decoy_calls=8),
        )

        assert results[0].other_tracks == 1
