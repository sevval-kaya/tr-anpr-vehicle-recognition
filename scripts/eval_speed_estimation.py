#!/usr/bin/env python
"""Evaluate the uncalibrated speed estimation (plaka.pipeline.speed,
docs/decisions.md #42) against real, physically-measured ground truth.

Protocol: mark two points a known distance apart (tape measure) on a
straight, flat path a real car or motorcycle can drive through — a
pedestrian/bicycle won't work, VehicleDetector only recognizes
car/motorcycle/bus/truck (configs/detection.yaml), so the pipeline never
even sees a walking test subject. Film one short clip per pass with the
same kind of camera setup the app would actually be used with (phone/
dashcam), and time each pass with a stopwatch: start the instant the
vehicle crosses the first mark, stop at the second. Ground truth speed
for that clip is distance_m / duration_seconds — no GPS/speedometer
needed. Repeat at a few different real speeds, distances, and camera
angles for an actual error *distribution*, not one anecdote — the known
error sources (camera-angle perspective, the assumed-vehicle-length
constant, tracker jitter — see decisions.md #42) are angle/distance
dependent, so a single pass can't represent them.

Manifest CSV columns (header required):

    video_path,distance_m,duration_seconds,rotate_degrees,notes
    clip_20kmh_front.mp4,15.0,2.7,0,
    clip_30kmh_angle.mp4,15.0,1.8,270,45 derece acidan

`rotate_degrees` and `notes` are optional (default 0 / empty) if the
column is omitted entirely; leave a cell blank to use the default for
just that row.

    python scripts/eval_speed_estimation.py data/external/speed_eval/manifest.csv

For each clip, the dominant track (the one tracked across the most
frames — assumed to be the real test vehicle, since a clean clip should
only really have one) is compared against that clip's ground truth.
Reports per-clip predicted vs. actual and the run's aggregate MAE/MAPE.
Writes a results CSV next to the manifest for pasting into a report.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2

from plaka.pipeline.builder import REPO_ROOT, build_pipeline_from_config
from plaka.pipeline.inference_pipeline import InferencePipeline
from plaka.pipeline.tracker import VehicleTracker, apply_consensus
from plaka.pipeline.video_io import FrameSamplingPlan, timestamp_seconds
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


@dataclass
class ClipResult:
    video_path: str
    ground_truth_kmh: float
    predicted_kmh: float | None
    vehicle_type: str | None
    tracked_frames: int
    other_tracks: int
    notes: str

    @property
    def abs_error_kmh(self) -> float | None:
        if self.predicted_kmh is None:
            return None
        return abs(self.predicted_kmh - self.ground_truth_kmh)

    @property
    def percent_error(self) -> float | None:
        if self.predicted_kmh is None or self.ground_truth_kmh == 0:
            return None
        return abs(self.predicted_kmh - self.ground_truth_kmh) / self.ground_truth_kmh * 100


@dataclass
class _TrackSnapshot:
    vehicle_type: str
    estimated_speed_kmh: float | None
    frames_tracked: int


def _process_clip(
    pipeline: InferencePipeline, video_path: Path, rotate_degrees: int
) -> tuple[float | None, str | None, int, int]:
    """Runs the real pipeline + tracker over one clip and returns the
    dominant track's (estimated_speed_kmh, vehicle_type, frames_tracked,
    other_significant_track_count) — the last one is a red flag: > 0
    means another vehicle was in frame long enough to plausibly be
    confused with the test subject, so the result is worth a second look.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    plan = FrameSamplingPlan.build(fps=fps, rotate_degrees=rotate_degrees)
    tracker = VehicleTracker()
    # Snapshotted after every frame (not read back from the tracker after
    # the loop) so a track that goes stale and gets retired before the
    # clip ends is still counted using its last-known state.
    track_snapshots: dict[int, _TrackSnapshot] = {}

    frame_index = 0
    try:
        while True:
            read_ok, frame = capture.read()
            if not read_ok:
                break
            frame = plan.prepare(frame)
            raw_result = pipeline.process_frame(frame, frame_index=frame_index)
            updated = apply_consensus(
                raw_result,
                tracker,
                frame_index,
                timestamp_seconds=timestamp_seconds(frame_index, fps),
            )
            for vehicle in updated.vehicles:
                if vehicle.track_id is None:
                    continue
                track = tracker.get_track(vehicle.track_id)
                if track is None:
                    continue
                track_snapshots[vehicle.track_id] = _TrackSnapshot(
                    vehicle_type=track.consensus_vehicle_type,
                    estimated_speed_kmh=track.estimated_speed_kmh,
                    frames_tracked=len(track.position_history),
                )
            frame_index += 1
    finally:
        capture.release()

    if not track_snapshots:
        return None, None, 0, 0

    dominant = max(track_snapshots.values(), key=lambda s: s.frames_tracked)
    other_tracks = sum(
        1 for s in track_snapshots.values() if s is not dominant and s.frames_tracked >= 5
    )
    return (
        dominant.estimated_speed_kmh,
        dominant.vehicle_type,
        dominant.frames_tracked,
        other_tracks,
    )


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {manifest_path}")
    return rows


def ground_truth_kmh(distance_m: float, duration_seconds: float) -> float:
    return distance_m / duration_seconds * 3.6


def evaluate(
    manifest_path: Path,
    output_csv: Path,
    pipeline_config_path: Path = DEFAULT_PIPELINE_CONFIG,
    pipeline: InferencePipeline | None = None,
) -> list[ClipResult]:
    """`pipeline` is only overridden by tests (a fake, model-free
    pipeline) — real runs always build the actual InferencePipeline from
    `pipeline_config_path`, same as every other entry point.
    """
    if pipeline is None:
        pipeline = build_pipeline_from_config(pipeline_config_path)
    manifest_dir = manifest_path.parent
    results: list[ClipResult] = []

    for row in _read_manifest(manifest_path):
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = manifest_dir / video_path
        distance_m = float(row["distance_m"])
        duration_seconds = float(row["duration_seconds"])
        rotate_degrees = int(row.get("rotate_degrees") or 0)
        notes = row.get("notes") or ""
        gt_kmh = ground_truth_kmh(distance_m, duration_seconds)

        if not video_path.exists():
            logger.warning("skipping missing clip: %s", video_path)
            continue

        predicted_kmh, vehicle_type, tracked_frames, other_tracks = _process_clip(
            pipeline, video_path, rotate_degrees
        )
        results.append(
            ClipResult(
                video_path=row["video_path"],
                ground_truth_kmh=gt_kmh,
                predicted_kmh=predicted_kmh,
                vehicle_type=vehicle_type,
                tracked_frames=tracked_frames,
                other_tracks=other_tracks,
                notes=notes,
            )
        )

    _report(results)
    _write_csv(results, output_csv)
    return results


def _report(results: list[ClipResult]) -> None:
    print(f"\n{'clip':<28} {'gerçek':>8} {'tahmin':>8} {'hata':>7} {'hata%':>7} "
          f"{'tip':>10} {'kare':>5} {'diğer?':>7}")
    print("-" * 90)
    for r in results:
        pred_str = f"{r.predicted_kmh:.1f}" if r.predicted_kmh is not None else "—"
        err_str = f"{r.abs_error_kmh:.1f}" if r.abs_error_kmh is not None else "—"
        pct_str = f"{r.percent_error:.0f}%" if r.percent_error is not None else "—"
        flag = f"{r.other_tracks} araç" if r.other_tracks else ""
        print(
            f"{r.video_path:<28} {r.ground_truth_kmh:>8.1f} {pred_str:>8} {err_str:>7} "
            f"{pct_str:>7} {(r.vehicle_type or '—'):>10} {r.tracked_frames:>5} {flag:>7}"
        )

    valid = [r for r in results if r.predicted_kmh is not None]
    missing = len(results) - len(valid)
    print("-" * 90)
    if not valid:
        print("Hiçbir klipte hız tahmin edilemedi (araç en az 2 kare boyunca takip edilmeliydi).")
        return

    abs_errors = [r.abs_error_kmh for r in valid if r.abs_error_kmh is not None]
    pct_errors = [r.percent_error for r in valid if r.percent_error is not None]
    print(
        f"{len(valid)}/{len(results)} klip değerlendirildi"
        + (f" ({missing} klipte tahmin çıkmadı)" if missing else "")
    )
    print(f"MAE  (ortalama mutlak hata):  {statistics.mean(abs_errors):.1f} km/h")
    print(f"MAPE (ortalama yüzde hata):   {statistics.mean(pct_errors):.1f}%")
    print(f"Medyan mutlak hata:            {statistics.median(abs_errors):.1f} km/h")
    print(f"En kötü tekil hata:             {max(abs_errors):.1f} km/h")
    flagged = sum(1 for r in valid if r.other_tracks)
    if flagged:
        print(
            f"\nUYARI: {flagged} klipte test aracı dışında en az 5 kare boyunca görünen "
            "başka bir araç da vardı — o klipler için 'baskın' track gerçekten test "
            "aracınız olmayabilir, tek tek kontrol edin."
        )


def _write_csv(results: list[ClipResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_path", "ground_truth_kmh", "predicted_kmh", "abs_error_kmh",
                "percent_error", "vehicle_type", "tracked_frames", "other_tracks", "notes",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.video_path,
                    f"{r.ground_truth_kmh:.2f}",
                    f"{r.predicted_kmh:.2f}" if r.predicted_kmh is not None else "",
                    f"{r.abs_error_kmh:.2f}" if r.abs_error_kmh is not None else "",
                    f"{r.percent_error:.1f}" if r.percent_error is not None else "",
                    r.vehicle_type or "",
                    r.tracked_frames,
                    r.other_tracks,
                    r.notes,
                ]
            )
    print(f"\nSonuçlar yazıldı: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="Path to the manifest CSV")
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write the results CSV (default: <manifest-dir>/speed_eval_results.csv)",
    )
    args = parser.parse_args()

    output_csv = args.output_csv or (args.manifest.parent / "speed_eval_results.csv")
    evaluate(args.manifest, output_csv, pipeline_config_path=args.pipeline_config)


if __name__ == "__main__":
    main()
