"""Background video-processing jobs for the web app.

Video inference is too slow to run inside a single HTTP request (a
30-second clip can take well over a minute on this hardware — see
docs/decisions.md #28/#29), so uploads are handed to a background thread
immediately and the client polls for progress instead of blocking on the
response. This mirrors the producer/consumer pattern already used for the
live camera in scripts/run_inference_video.py, just applied to "a whole
video" instead of "one frame at a time". Rotation and time-based frame
sampling use the same plaka.pipeline.video_io helpers as that script, so
both code paths stay in sync (docs/decisions.md).

Every tracked vehicle (plaka.pipeline.tracker — cross-frame consensus
voting) gets one gallery entry, plate read or not — vehicle_type_counts
and vehicle_sightings are both keyed by track, not by raw per-frame
detections, so a car visible across 400 frames counts once, not 400
times (docs/decisions.md #39/#40).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2

from plaka.pipeline.inference_pipeline import InferencePipeline
from plaka.pipeline.schemas import FrameResult
from plaka.pipeline.speed import exceeds_speed_limit
from plaka.pipeline.tracker import VehicleTracker, apply_consensus
from plaka.pipeline.video_io import FrameSamplingPlan, rotates_dimensions, timestamp_seconds
from plaka.pipeline.visualization import annotate_frame
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

JobStatus = Literal["queued", "processing", "done", "error"]
PlateStatus = Literal["read", "unreadable", "no_plate"]

# City-street ballpark default (docs/decisions.md #42) — always overridable
# per request/job, this is only the value pre-filled in the UI.
DEFAULT_SPEED_LIMIT_KMH = 50.0

# Cap on how many distinct *vehicles* (tracks) a job keeps a card for —
# every tracked vehicle counts now, not just the ones with a successful
# plate read (docs/decisions.md #40), so this needs more headroom than
# before (arac3.mp4 alone has ~28 real tracks across 901 frames of
# ordinary street traffic); still bounded so a very long video doesn't
# write hundreds of thumbnail JPEGs.
MAX_TRACKED_VEHICLES = 40


@dataclass
class VehicleSighting:
    frame_index: int
    timestamp_seconds: float
    vehicle_type: str
    plate_status: PlateStatus
    plate_text: str | None
    raw_ocr_text: str | None
    observation_count: int
    thumbnail_url: str
    estimated_speed_kmh: float | None
    speed_limit_exceeded: bool


def _dedupe_sightings(track_sightings: dict[int, VehicleSighting]) -> list[VehicleSighting]:
    """The tracker can end up representing one physical vehicle as more
    than one track (a brief occlusion/gap the greedy IoU matcher didn't
    bridge, or the vehicle leaving and re-entering frame) — see
    docs/decisions.md #37 for the same issue on the type-vote side.
    Normalized plate text is a reliable dedupe key for anything with a
    "read" status: if two tracks both settled on the same plate, they're
    almost certainly the same car, so only the better-established one
    (more observations) is kept. Tracks with no readable plate have no
    such key — perfect dedupe isn't possible there, so they're kept as
    separate cards (see docs/decisions.md #40).

    Returns read-status entries first, then the rest, per how the
    gallery is meant to be scanned (docs/decisions.md #40).
    """
    best_by_text: dict[str, VehicleSighting] = {}
    others: list[VehicleSighting] = []
    for sighting in track_sightings.values():
        if sighting.plate_status == "read" and sighting.plate_text:
            existing = best_by_text.get(sighting.plate_text)
            if existing is None or sighting.observation_count > existing.observation_count:
                best_by_text[sighting.plate_text] = sighting
        else:
            others.append(sighting)
    read_sightings = sorted(best_by_text.values(), key=lambda s: s.frame_index)
    return read_sightings + others


@dataclass
class VideoJob:
    job_id: str
    original_filename: str
    input_path: Path
    output_dir: Path
    rotate_degrees: int = 0
    sample_interval_seconds: float | None = None
    speed_limit_kmh: float = DEFAULT_SPEED_LIMIT_KMH
    status: JobStatus = "queued"
    frames_processed: int = 0
    total_frames: int = 0
    frame_stride: int = 1
    error_message: str | None = None
    vehicle_type_counts: dict[str, int] = field(default_factory=dict)
    vehicle_sightings: list[VehicleSighting] = field(default_factory=list)
    output_video_path: Path | None = None
    estimated_seconds_remaining: float | None = None

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.frames_processed / self.total_frames)


class JobManager:
    """Owns the in-memory job registry and runs each job's video processing
    on its own daemon thread. Deliberately process-local, in-memory state
    (no database) — this is a single-machine demo app, not a deployed
    service; a restart losing job history is an acceptable trade for the
    simplicity.
    """

    def __init__(
        self,
        pipeline: InferencePipeline,
        jobs_root: Path,
        inference_lock: threading.Lock | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._jobs_root = jobs_root
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.Lock()
        # Shared with the FastAPI app's image/camera routes (see
        # plaka.web.app._infer_and_annotate) — serializes access to the
        # pipeline's model objects across threads, since this job runs on
        # its own background thread while those run via asyncio.to_thread
        # on the request thread pool, all against the same pipeline instance.
        self._inference_lock = inference_lock or threading.Lock()

    def submit(
        self,
        video_bytes: bytes,
        original_filename: str,
        rotate_degrees: int = 0,
        sample_interval_seconds: float | None = None,
        speed_limit_kmh: float = DEFAULT_SPEED_LIMIT_KMH,
    ) -> str:
        job_id = uuid.uuid4().hex
        output_dir = self._jobs_root / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = output_dir / f"input{Path(original_filename).suffix or '.mp4'}"
        input_path.write_bytes(video_bytes)

        job = VideoJob(
            job_id=job_id,
            original_filename=original_filename,
            input_path=input_path,
            output_dir=output_dir,
            rotate_degrees=rotate_degrees,
            sample_interval_seconds=sample_interval_seconds,
            speed_limit_kmh=speed_limit_kmh,
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        assert job is not None
        job.status = "processing"
        start = time.monotonic()

        try:
            capture = cv2.VideoCapture(str(job.input_path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open uploaded video: {job.original_filename!r}")

            job.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if rotates_dimensions(job.rotate_degrees):
                width, height = height, width

            plan = FrameSamplingPlan.build(
                fps=fps,
                frame_stride=1,
                sample_interval_seconds=job.sample_interval_seconds,
                rotate_degrees=job.rotate_degrees,
            )
            job.frame_stride = plan.frame_stride
            logger.info(
                "job %s: fps=%.1f rotate=%d sample_interval=%s -> frame_stride=%d",
                job_id, fps, job.rotate_degrees, job.sample_interval_seconds, plan.frame_stride,
            )

            output_path = job.output_dir / "annotated.mp4"
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            tracker = VehicleTracker()
            track_thumb_index: dict[int, int] = {}
            track_types: dict[int, str] = {}
            # Representative-frame selection, per track: once a track has
            # ever had a plate box, the best (highest OCR confidence) such
            # frame wins, regardless of vehicle-box size — that's the frame
            # most likely to show a legible plate crop. Tracks that never
            # get a plate box fall back to their largest-vehicle-box frame
            # instead (best chance a human could still spot the plate).
            track_has_plate_box: dict[int, bool] = {}
            track_best_plate_score: dict[int, float] = {}
            track_best_area: dict[int, float] = {}
            track_best_raw_text: dict[int, str] = {}
            track_sightings: dict[int, VehicleSighting] = {}
            frame_index = 0
            last_result: FrameResult | None = None
            processed_frame_seconds: list[float] = []
            try:
                while True:
                    read_ok, frame = capture.read()
                    if not read_ok:
                        break
                    frame = plan.prepare(frame)

                    if plan.should_process(frame_index):
                        frame_start = time.monotonic()
                        with self._inference_lock:
                            raw_result = self._pipeline.process_frame(frame, frame_index=frame_index)
                        last_result = apply_consensus(
                            raw_result,
                            tracker,
                            frame_index,
                            timestamp_seconds=timestamp_seconds(frame_index, fps),
                        )
                        processed_frame_seconds.append(time.monotonic() - frame_start)
                        # Simple moving average over the last 20 processed
                        # frames — recent speed predicts remaining speed
                        # better than an all-time average would after a
                        # slow model-warmup first frame.
                        recent = processed_frame_seconds[-20:]
                        avg_seconds = sum(recent) / len(recent)
                        remaining_processed_frames = max(
                            0, (job.total_frames - frame_index) // plan.frame_stride
                        )
                        job.estimated_seconds_remaining = avg_seconds * remaining_processed_frames

                    result = last_result
                    annotated = (
                        annotate_frame(frame, result, speed_limit_kmh=job.speed_limit_kmh)
                        if result is not None
                        else frame
                    )
                    writer.write(annotated)

                    if result is not None:
                        for vehicle in result.vehicles:
                            if vehicle.track_id is None:
                                continue
                            track_id = vehicle.track_id

                            # vehicle_type here is already the track's
                            # consensus type (plaka.pipeline.tracker), so
                            # upserting on every sighting and recomputing
                            # the histogram from the current dict keeps
                            # this correct even if the consensus shifts
                            # mid-video (docs/decisions.md #39).
                            track_types[track_id] = vehicle.vehicle_type
                            job.vehicle_type_counts = dict(Counter(track_types.values()))

                            plate = vehicle.plate
                            box_area = vehicle.box.width * vehicle.box.height

                            should_update_thumb = False
                            if plate is not None:
                                track_has_plate_box[track_id] = True
                                prior_score = track_best_plate_score.get(track_id)
                                if prior_score is None or plate.ocr_confidence > prior_score:
                                    track_best_plate_score[track_id] = plate.ocr_confidence
                                    should_update_thumb = True
                                    if plate.raw_text:
                                        track_best_raw_text[track_id] = plate.raw_text
                            elif track_id not in track_best_plate_score:
                                prior_area = track_best_area.get(track_id)
                                if prior_area is None or box_area > prior_area:
                                    track_best_area[track_id] = box_area
                                    should_update_thumb = True

                            if track_id not in track_thumb_index:
                                if len(track_thumb_index) >= MAX_TRACKED_VEHICLES:
                                    continue
                                track_thumb_index[track_id] = len(track_thumb_index)

                            if should_update_thumb:
                                thumb_name = f"vehicle_{track_thumb_index[track_id]:02d}.jpg"
                                cv2.imwrite(str(job.output_dir / thumb_name), annotated)

                            track = tracker.get_track(track_id)
                            consensus_text = track.consensus_text if track is not None else None
                            plate_status: PlateStatus
                            if consensus_text is not None:
                                plate_status = "read"
                            elif track_has_plate_box.get(track_id):
                                plate_status = "unreadable"
                            else:
                                plate_status = "no_plate"

                            track_sightings[track_id] = VehicleSighting(
                                frame_index=frame_index,
                                timestamp_seconds=timestamp_seconds(frame_index, fps),
                                vehicle_type=vehicle.vehicle_type,
                                plate_status=plate_status,
                                plate_text=consensus_text,
                                raw_ocr_text=track_best_raw_text.get(track_id),
                                observation_count=track.observation_count if track is not None else 0,
                                thumbnail_url=(
                                    f"/api/jobs/{job_id}/thumbnail/"
                                    f"vehicle_{track_thumb_index[track_id]:02d}.jpg"
                                ),
                                estimated_speed_kmh=vehicle.estimated_speed_kmh,
                                speed_limit_exceeded=exceeds_speed_limit(
                                    vehicle.estimated_speed_kmh, job.speed_limit_kmh
                                ),
                            )
                            job.vehicle_sightings = _dedupe_sightings(track_sightings)

                    frame_index += 1
                    job.frames_processed = frame_index
            finally:
                capture.release()
                writer.release()

            job.output_video_path = output_path
            job.estimated_seconds_remaining = 0.0
            job.status = "done"
            read_count = sum(1 for s in job.vehicle_sightings if s.plate_status == "read")
            logger.info(
                "job %s done: %d frame(s) (%d actually processed), %d tracked vehicle(s) "
                "(%d with a valid plate read), %.1fs",
                job_id,
                frame_index,
                len(processed_frame_seconds),
                len(job.vehicle_sightings),
                read_count,
                time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001 - reported to the client, not swallowed
            logger.exception("job %s failed", job_id)
            job.status = "error"
            job.error_message = str(exc)
