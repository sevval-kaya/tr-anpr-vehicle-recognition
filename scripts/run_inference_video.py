#!/usr/bin/env python
"""Run InferencePipeline on a video file or live camera, frame by frame,
and display and/or save the annotated result.

    python scripts/run_inference_video.py 0                    # webcam, live window
    python scripts/run_inference_video.py path/to/clip.mp4
    python scripts/run_inference_video.py path/to/clip.mp4 --output outputs/annotated.mp4
    python scripts/run_inference_video.py 0 --no-display --output outputs/annotated.mp4
    python scripts/run_inference_video.py path/to/clip.mp4 --rotate 270 --sample-interval-seconds 1

Each processed frame still runs through InferencePipeline independently,
but a simple cross-frame vehicle tracker (plaka.pipeline.tracker, greedy
IoU matching) now sits on top: the same vehicle's plate readings across
several frames are combined by majority vote into one consensus string,
which is what gets displayed/logged instead of that frame's own (often
noisier) single-frame read — see docs/decisions.md #35. Make/model
classification is disabled by default (see docs/decisions.md #29) — each
vehicle box is labeled with its detected type (car/motorcycle/bus/truck)
and its (consensus) read plate, not a brand guess.

Rotation (--rotate 90/180/270) is opt-in, not auto-detected: video-container
orientation metadata proved unreliable on this project's own test clips —
one already-correct clip reports a stray 180° tag, one genuinely-rotated
clip reports none at all — so guessing from metadata would silently break
one or the other. Pick the value that makes the output look upright; 0
(default) leaves frames untouched. See docs/decisions.md.

Camera sources (source is a bare integer, e.g. "0") are read on a
background capture thread by default so a slow pipeline can never make the
live view fall behind/stutter waiting on cv2.VideoCapture.read() — the
processing loop always grabs whichever frame is newest, skipping ones it
can't keep up with rather than queuing them (see ThreadedFrameGrabber
below, docs/decisions.md #29). Pass --no-threaded to fall back to the
strictly sequential capture->process->display loop for comparison. Video
files always use the sequential loop (there's no live "falling behind" to
fix — --frame-stride/--sample-interval-seconds/--max-frames already
control how much work is done).

Press 'q' in the display window to stop early.
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from plaka.pipeline.builder import REPO_ROOT, build_pipeline_from_config
from plaka.pipeline.inference_pipeline import InferencePipeline
from plaka.pipeline.schemas import FrameResult
from plaka.pipeline.tracker import VehicleTracker, apply_consensus
from plaka.pipeline.video_io import (
    FrameSamplingPlan,
    apply_rotation,
    rotates_dimensions,
    timestamp_seconds,
)
from plaka.pipeline.visualization import annotate_frame
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
WINDOW_TITLE = "plaka - InferencePipeline"
FPS_LOG_INTERVAL = 30

# A frame-to-frame processing gap more than this many times the median is
# reported as a "latency spike" (visible stutter), not just normal jitter.
SPIKE_MEDIAN_MULTIPLIER = 3.0

# Idle poll interval while waiting for a new frame from the background
# capture thread (threaded camera mode only) — short enough not to add
# perceptible delay, long enough to not busy-spin a full CPU core.
_POLL_INTERVAL_SECONDS = 0.001


def _is_camera_source(source: str) -> bool:
    return source.isdigit()


def _open_capture(source: str) -> cv2.VideoCapture:
    # A bare integer string ("0", "1", ...) is a camera index; anything
    # else is treated as a file path.
    capture_source: int | str = int(source) if _is_camera_source(source) else source
    return cv2.VideoCapture(capture_source)


def _open_writer(
    output_path: Path, capture: cv2.VideoCapture, rotate_degrees: int
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotates_dimensions(rotate_degrees):
        width, height = height, width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def _log_summary(frame_index: int, fps: float, result: FrameResult) -> None:
    t = timestamp_seconds(frame_index, fps)
    for vehicle in result.vehicles:
        if vehicle.plate is not None and vehicle.plate.is_format_valid:
            logger.info(
                "frame %d (t=%.2fs): type=%s plate=%s",
                frame_index,
                t,
                vehicle.vehicle_type,
                vehicle.plate.normalized_text,
            )


class LatencyTracker:
    """Records the wall-clock gap between consecutive *processed* frames
    (i.e. calls to InferencePipeline.process_frame, not every capture
    thread tick) and summarizes it as p50/p95/max plus a stutter count —
    a plain average fps hides exactly the kind of occasional multi-second
    stall a user would perceive as "kasma" (see docs/decisions.md #29).
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._last_timestamp: float | None = None
        self.deltas_seconds: list[float] = []

    def mark(self) -> None:
        now = time.monotonic()
        if self._last_timestamp is not None:
            self.deltas_seconds.append(now - self._last_timestamp)
        self._last_timestamp = now

    def report(self) -> None:
        frame_count = len(self.deltas_seconds) + 1 if self.deltas_seconds else 0
        if len(self.deltas_seconds) < 2:
            logger.info(
                "%s: only %d processed frame(s) — not enough to report latency stats",
                self.label,
                frame_count,
            )
            return

        deltas_ms = sorted(d * 1000 for d in self.deltas_seconds)
        p50 = statistics.median(deltas_ms)
        p95_index = min(len(deltas_ms) - 1, round(0.95 * (len(deltas_ms) - 1)))
        p95 = deltas_ms[p95_index]
        worst = max(deltas_ms)
        spike_threshold = p50 * SPIKE_MEDIAN_MULTIPLIER
        spikes = sum(1 for d in deltas_ms if d > spike_threshold)

        logger.info(
            "%s: %d processed frame(s) | frame-to-frame latency p50=%.1fms p95=%.1fms "
            "max=%.1fms | %d spike(s) (>%.1fms, %.0fx median)",
            self.label,
            frame_count,
            p50,
            p95,
            worst,
            spikes,
            spike_threshold,
            SPIKE_MEDIAN_MULTIPLIER,
        )


class ThreadedFrameGrabber:
    """Reads frames from `capture` continuously on a background thread into
    a single-slot buffer, so a slow consumer can never make the camera
    read itself block or backlog. The consumer (`read_latest`) always gets
    whichever frame is newest — if it can't keep up, it skips frames
    instead of falling behind a growing queue of stale ones. This is what
    decouples the live display from pipeline speed (docs/decisions.md #29).
    """

    def __init__(self, capture: cv2.VideoCapture, rotate_degrees: int = 0) -> None:
        self._capture = capture
        self._rotate_degrees = rotate_degrees
        self._lock = threading.Lock()
        self._frame: NDArray[np.uint8] | None = None
        self._generation = 0
        self._read_failed = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> ThreadedFrameGrabber:
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            read_ok, frame = self._capture.read()
            if not read_ok:
                with self._lock:
                    self._read_failed = True
                return
            if self._rotate_degrees:
                frame = apply_rotation(frame, self._rotate_degrees)
            with self._lock:
                self._frame = frame
                self._generation += 1

    def read_latest(self) -> tuple[NDArray[np.uint8] | None, int, bool]:
        """Returns (latest_frame_or_None, generation_counter, capture_ended)."""
        with self._lock:
            return self._frame, self._generation, self._read_failed

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def _process_and_annotate(
    pipeline: InferencePipeline,
    tracker: VehicleTracker,
    frame: NDArray[np.uint8],
    frame_index: int,
    fps: float,
    plan: FrameSamplingPlan,
    last_result: FrameResult | None,
    latency: LatencyTracker,
    timestamp_seconds_value: float,
    speed_limit_kmh: float,
) -> tuple[FrameResult | None, NDArray[np.uint8]]:
    if plan.should_process(frame_index):
        raw_result = pipeline.process_frame(frame, frame_index=frame_index)
        latency.mark()
        last_result = apply_consensus(
            raw_result, tracker, frame_index, timestamp_seconds=timestamp_seconds_value
        )
        _log_summary(frame_index, fps, last_result)
    annotated = (
        annotate_frame(frame, last_result, speed_limit_kmh=speed_limit_kmh)
        if last_result is not None
        else frame
    )
    return last_result, annotated


def _run_sequential(
    capture: cv2.VideoCapture,
    pipeline: InferencePipeline,
    writer: cv2.VideoWriter | None,
    display: bool,
    plan: FrameSamplingPlan,
    max_frames: int | None,
    speed_limit_kmh: float,
) -> LatencyTracker:
    latency = LatencyTracker("sequential")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = VehicleTracker()
    frame_index = 0
    last_result: FrameResult | None = None
    start_time = time.monotonic()

    while True:
        read_ok, frame = capture.read()
        if not read_ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        frame = plan.prepare(frame)

        # A video file's frame_index/fps ratio is a reliable time base
        # (unlike live camera — see _run_threaded below and
        # docs/decisions.md #42).
        last_result, annotated = _process_and_annotate(
            pipeline, tracker, frame, frame_index, fps, plan, last_result, latency,
            timestamp_seconds_value=timestamp_seconds(frame_index, fps),
            speed_limit_kmh=speed_limit_kmh,
        )

        if writer is not None:
            writer.write(annotated)
        if display:
            cv2.imshow(WINDOW_TITLE, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("stopped by user ('q')")
                break

        frame_index += 1
        if frame_index % FPS_LOG_INTERVAL == 0:
            elapsed = time.monotonic() - start_time
            overall_fps = frame_index / elapsed if elapsed > 0 else 0.0
            logger.info("frame %d | %.1f fps overall", frame_index, overall_fps)

    logger.info("done: %d frame(s) processed", frame_index)
    return latency


def _run_threaded(
    capture: cv2.VideoCapture,
    pipeline: InferencePipeline,
    writer: cv2.VideoWriter | None,
    display: bool,
    plan: FrameSamplingPlan,
    max_frames: int | None,
    speed_limit_kmh: float,
) -> LatencyTracker:
    latency = LatencyTracker("threaded")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = VehicleTracker()
    grabber = ThreadedFrameGrabber(capture, rotate_degrees=plan.rotate_degrees).start()

    frame_index = 0
    last_seen_generation = -1
    last_result: FrameResult | None = None
    last_annotated: NDArray[np.uint8] | None = None
    start_time = time.monotonic()

    try:
        while True:
            frame, generation, capture_ended = grabber.read_latest()

            if frame is None:
                if capture_ended:
                    break
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            if generation == last_seen_generation:
                # No new frame since last loop; redraw the last annotated
                # result so the window stays responsive without touching
                # the pipeline or the capture thread.
                if display and last_annotated is not None:
                    cv2.imshow(WINDOW_TITLE, last_annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("stopped by user ('q')")
                        break
                else:
                    time.sleep(_POLL_INTERVAL_SECONDS)
                if capture_ended:
                    break
                continue
            last_seen_generation = generation

            if max_frames is not None and frame_index >= max_frames:
                break

            # Rotation is already applied inside the grabber thread (it has
            # to be, so ThreadedFrameGrabber's own frame buffer stays
            # consistent regardless of who reads it next). Real wall-clock
            # time, not frame_index/fps: a live camera's actual frame
            # interval isn't constant (grabber skips frames the pipeline
            # can't keep up with — see ThreadedFrameGrabber docstring and
            # docs/decisions.md #42), so fps would be the wrong time base.
            last_result, annotated = _process_and_annotate(
                pipeline, tracker, frame, frame_index, fps, plan, last_result, latency,
                timestamp_seconds_value=time.monotonic(),
                speed_limit_kmh=speed_limit_kmh,
            )
            last_annotated = annotated

            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow(WINDOW_TITLE, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("stopped by user ('q')")
                    break

            frame_index += 1
            if frame_index % FPS_LOG_INTERVAL == 0:
                elapsed = time.monotonic() - start_time
                overall_fps = frame_index / elapsed if elapsed > 0 else 0.0
                logger.info("frame %d | %.1f fps overall", frame_index, overall_fps)
    finally:
        _, final_generation, _ = grabber.read_latest()
        grabber.stop()

    # The generation counter increments once per frame the capture thread
    # actually grabbed, independent of how many the pipeline kept up with —
    # the gap between the two is direct evidence that capture never blocked
    # on processing (the mechanism this mode exists for), not just a speed
    # number. If pipeline throughput >= camera fps, this ratio is ~1:1 and
    # threading buys nothing extra; the gap only opens once processing falls
    # behind the camera, which is exactly the scenario --no-threaded stalls on.
    skipped = max(0, final_generation - frame_index)
    logger.info(
        "capture thread grabbed %d frame(s) total, %d were processed (%d skipped "
        "while the pipeline was still busy on the previous one)",
        final_generation,
        frame_index,
        skipped,
    )

    logger.info("done: %d frame(s) processed", frame_index)
    return latency


def run(
    source: str,
    pipeline_config_path: Path,
    output_path: Path | None,
    display: bool,
    frame_stride: int,
    sample_interval_seconds: float | None,
    rotate_degrees: int,
    max_frames: int | None,
    no_threaded: bool,
    speed_limit_kmh: float,
) -> None:
    capture = _open_capture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video source: {source!r}")

    writer = _open_writer(output_path, capture, rotate_degrees) if output_path is not None else None
    pipeline = build_pipeline_from_config(pipeline_config_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    plan = FrameSamplingPlan.build(
        fps=fps,
        frame_stride=frame_stride,
        sample_interval_seconds=sample_interval_seconds,
        rotate_degrees=rotate_degrees,
    )
    if sample_interval_seconds is not None and frame_stride != 1:
        logger.info(
            "--sample-interval-seconds=%.2f given -> overrides --frame-stride=%d "
            "(effective frame_stride=%d at %.1f fps)",
            sample_interval_seconds,
            frame_stride,
            plan.frame_stride,
            fps,
        )
    elif sample_interval_seconds is not None:
        logger.info(
            "--sample-interval-seconds=%.2f -> effective frame_stride=%d at %.1f fps",
            sample_interval_seconds,
            plan.frame_stride,
            fps,
        )

    use_threaded = _is_camera_source(source) and not no_threaded
    logger.info(
        "source=%r mode=%s rotate=%d", source, "threaded (camera)" if use_threaded else "sequential", rotate_degrees
    )

    try:
        if use_threaded:
            latency = _run_threaded(
                capture, pipeline, writer, display, plan, max_frames, speed_limit_kmh
            )
        else:
            latency = _run_sequential(
                capture, pipeline, writer, display, plan, max_frames, speed_limit_kmh
            )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    latency.report()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="Camera index (e.g. 0) or path to a video file")
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    parser.add_argument("--output", type=Path, default=None, help="Write annotated video here")
    parser.add_argument("--no-display", action="store_true", help="Don't open a live preview window")
    parser.add_argument(
        "--no-threaded",
        action="store_true",
        help="Force the sequential capture->process->display loop even for a camera "
        "source, for comparison against the default threaded mode.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate every frame clockwise by this many degrees before anything else runs "
        "(0=off). Not auto-detected from video metadata — see module docstring.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run the pipeline every Nth frame (default 1 = every frame); "
        "skipped frames re-draw the last computed result so playback doesn't flicker. "
        "Ignored if --sample-interval-seconds is given.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=None,
        help="Run the pipeline roughly once every N seconds of video, regardless of the "
        "source's fps (frame_stride = round(fps * N)). Overrides --frame-stride.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument(
        "--speed-limit-kmh",
        type=float,
        default=50.0,
        help="Uncalibrated speed estimate above this (km/h) is drawn in red below the "
        "vehicle box instead of white (docs/decisions.md #42). Default matches the web "
        "app's default (configs/pipeline.yaml speed.default_speed_limit_kmh).",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.speed_limit_kmh <= 0:
        parser.error("--speed-limit-kmh must be > 0")
    if args.sample_interval_seconds is not None and args.sample_interval_seconds <= 0:
        parser.error("--sample-interval-seconds must be > 0")

    run(
        source=args.source,
        pipeline_config_path=args.pipeline_config,
        output_path=args.output,
        display=not args.no_display,
        frame_stride=args.frame_stride,
        sample_interval_seconds=args.sample_interval_seconds,
        rotate_degrees=args.rotate,
        max_frames=args.max_frames,
        no_threaded=args.no_threaded,
        speed_limit_kmh=args.speed_limit_kmh,
    )


if __name__ == "__main__":
    main()
