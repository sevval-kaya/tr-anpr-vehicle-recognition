import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from plaka.pipeline.schemas import BoundingBox, FrameResult, PlateReading, VehicleDetection
from plaka.web.app import create_app


@dataclass
class FakePipeline:
    """Returns a fixed FrameResult regardless of input — the web layer
    only needs to be tested against the InferencePipeline *interface*
    (one process_frame(frame, frame_index=0) call), not real models.
    """

    result: FrameResult
    calls: list[int] = field(default_factory=list)
    frame_shapes: list[tuple[int, ...]] = field(default_factory=list)

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        self.calls.append(frame_index)
        self.frame_shapes.append(frame.shape)
        return self.result


def _one_vehicle_result() -> FrameResult:
    return FrameResult(
        frame_index=0,
        vehicles=[
            VehicleDetection(
                box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
                vehicle_type="car",
                detection_confidence=0.91,
                plate=PlateReading(
                    box=BoundingBox(x_min=20, y_min=80, x_max=80, y_max=95),
                    raw_text="34AB123",
                    normalized_text="34 AB 123",
                    is_format_valid=True,
                    detection_confidence=0.9,
                    ocr_confidence=0.95,
                ),
            )
        ],
    )


def _jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return bytes(buffer)


def _tiny_mp4_bytes(tmp_path: Path, frame_count: int = 5, fps: float = 10.0) -> bytes:
    path = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(frame_count):
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def _make_client(tmp_path: Path, result: FrameResult | None = None) -> tuple[TestClient, FakePipeline]:
    pipeline = FakePipeline(result=result if result is not None else _one_vehicle_result())
    app = create_app(pipeline=pipeline, jobs_root=tmp_path / "jobs")
    return TestClient(app), pipeline


def test_index_serves_the_frontend(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.get("/")
    assert response.status_code == 200
    assert "Plaka Tanıma Sistemi" in response.text


def test_infer_image_returns_annotated_result_and_vehicle_summary(tmp_path: Path) -> None:
    client, pipeline = _make_client(tmp_path)

    response = client.post(
        "/api/infer/image", files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["vehicles"]) == 1
    vehicle = data["vehicles"][0]
    assert vehicle["vehicle_type"] == "car"
    assert vehicle["plate_text"] == "34 AB 123"
    assert vehicle["plate_valid"] is True
    assert data["annotated_image_base64"]  # non-empty
    assert pipeline.calls == [0]


def test_infer_image_rejects_undecodable_upload(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.post(
        "/api/infer/image", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
    )
    assert response.status_code == 400


def test_infer_video_job_runs_to_completion_and_records_plate(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=5)

    submit = client.post(
        "/api/infer/video", files={"file": ("clip.mp4", video_bytes, "video/mp4")}
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)

    assert job is not None
    assert job["status"] == "done", job.get("error_message")
    assert job["frames_processed"] == 5
    assert job["frame_stride"] == 1  # no sample_interval_seconds given -> every frame
    # Same vehicle box every frame -> one track, counted once, not once per
    # frame it appears in (see docs/decisions.md #39).
    assert job["vehicle_type_counts"] == {"car": 1}
    assert len(job["vehicle_sightings"]) == 1
    assert job["vehicle_sightings"][0]["plate_status"] == "read"
    assert job["vehicle_sightings"][0]["plate_text"] == "34 AB 123"
    # FakePipeline returns the identical result every frame, so all 5 frames
    # match the same track (see plaka.pipeline.tracker) and the sighting is
    # continuously upserted — the final value reflects the LAST frame that
    # updated it (frame 4 of 5, at 10fps), not the first sighting.
    assert job["vehicle_sightings"][0]["frame_index"] == 4
    assert job["vehicle_sightings"][0]["timestamp_seconds"] == 0.4
    assert job["video_url"] == f"/api/jobs/{job_id}/video"

    video_response = client.get(job["video_url"])
    assert video_response.status_code == 200
    assert video_response.content  # non-empty file

    thumb_url = job["vehicle_sightings"][0]["thumbnail_url"]
    thumb_response = client.get(thumb_url)
    assert thumb_response.status_code == 200


def test_infer_video_rejects_invalid_rotate(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=2)
    response = client.post(
        "/api/infer/video",
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        data={"rotate": "45"},
    )
    assert response.status_code == 400


def test_infer_video_rotate_90_swaps_output_dimensions_and_frames_fed_to_pipeline(
    tmp_path: Path,
) -> None:
    client, pipeline = _make_client(tmp_path)
    # source frames are 64 wide x 48 tall; rotated 90 they must become 48x64.
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=2)

    submit = client.post(
        "/api/infer/video",
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        data={"rotate": "90"},
    )
    job_id = submit.json()["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job is not None and job["status"] == "done", job and job.get("error_message")

    # Every frame handed to the pipeline should already be rotated: 48 tall x 64 wide -> 64x48.
    assert all(shape[:2] == (64, 48) for shape in pipeline.frame_shapes)

    video_bytes_out = client.get(job["video_url"]).content
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(video_bytes_out)
    capture = cv2.VideoCapture(str(out_path))
    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (48, 64)
    capture.release()


def test_infer_video_sample_interval_seconds_reduces_pipeline_calls(tmp_path: Path) -> None:
    client, pipeline = _make_client(tmp_path)
    # 20 frames at 10fps = 2 seconds of video; sampling once a second should
    # only run the pipeline on ~2 frames, not all 20 — even though all 20
    # still get read/written (skipped ones just reuse the last result).
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=20, fps=10.0)

    submit = client.post(
        "/api/infer/video",
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        data={"sample_interval_seconds": "1.0"},
    )
    job_id = submit.json()["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job is not None and job["status"] == "done", job and job.get("error_message")

    assert job["frames_processed"] == 20  # every frame still read/written
    assert job["frame_stride"] == 10  # round(10fps * 1.0s)
    assert len(pipeline.calls) == 2  # only frames 0 and 10 actually hit the model


def test_infer_video_rejects_non_positive_sample_interval(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=2)
    response = client.post(
        "/api/infer/video",
        files={"file": ("clip.mp4", video_bytes, "video/mp4")},
        data={"sample_interval_seconds": "0"},
    )
    assert response.status_code == 400


@dataclass
class _NoisyReadingPipeline:
    """Same vehicle box every frame (so the tracker treats it as one
    vehicle), but a different plate reading each time — simulates the real
    frame-to-frame OCR noise this test is meant to exercise (see
    docs/decisions.md #33/#35).
    """

    readings: list[str]

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        text = self.readings[frame_index % len(self.readings)]
        box = BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100)
        return FrameResult(
            frame_index=frame_index,
            vehicles=[
                VehicleDetection(
                    box=box,
                    vehicle_type="car",
                    detection_confidence=0.9,
                    plate=PlateReading(
                        box=BoundingBox(x_min=20, y_min=80, x_max=80, y_max=95),
                        raw_text=text.replace(" ", ""),
                        normalized_text=text,
                        is_format_valid=True,
                        detection_confidence=0.9,
                        ocr_confidence=0.7,
                    ),
                )
            ],
        )


def test_infer_video_collapses_noisy_frame_readings_into_one_consensus_sighting(
    tmp_path: Path,
) -> None:
    # 3 frames read as "23 AI 638", 1 as "23 MEL 638" — same physical
    # vehicle (same box every frame). Before cross-frame consensus this
    # produced 2 separate "unique" sightings; now it must produce exactly
    # one, holding the majority-vote text.
    pipeline = _NoisyReadingPipeline(readings=["23 AI 638", "23 AI 638", "23 MEL 638", "23 AI 638"])
    app = create_app(pipeline=pipeline, jobs_root=tmp_path / "jobs")
    client = TestClient(app)
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=4)

    submit = client.post("/api/infer/video", files={"file": ("clip.mp4", video_bytes, "video/mp4")})
    job_id = submit.json()["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)

    assert job is not None and job["status"] == "done", job and job.get("error_message")
    assert len(job["vehicle_sightings"]) == 1  # one vehicle, not one per distinct string
    assert job["vehicle_sightings"][0]["plate_text"] == "23 AI 638"  # majority (3 of 4)


@dataclass
class _UnreadablePlatePipeline:
    """A plate box is found every frame, but it never validates — the
    "araç var, plaka okunamıyor" case (docs/decisions.md #40)."""

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        return FrameResult(
            frame_index=frame_index,
            vehicles=[
                VehicleDetection(
                    box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
                    vehicle_type="car",
                    detection_confidence=0.9,
                    plate=PlateReading(
                        box=BoundingBox(x_min=20, y_min=80, x_max=80, y_max=95),
                        raw_text="GARBLED",
                        normalized_text=None,
                        is_format_valid=False,
                        detection_confidence=0.9,
                        ocr_confidence=0.3,
                    ),
                )
            ],
        )


@dataclass
class _NoPlateBoxPipeline:
    """A vehicle is detected but the plate detector never finds a box for
    it at all — distinct from "found a box but couldn't read it" (see
    docs/decisions.md #40's three-way plate_status)."""

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        return FrameResult(
            frame_index=frame_index,
            vehicles=[
                VehicleDetection(
                    box=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=100),
                    vehicle_type="truck",
                    detection_confidence=0.9,
                    plate=None,
                )
            ],
        )


def test_infer_video_vehicle_sightings_include_unreadable_and_no_plate_vehicles(
    tmp_path: Path,
) -> None:
    for pipeline, expected_status, expected_raw in [
        (_UnreadablePlatePipeline(), "unreadable", "GARBLED"),
        (_NoPlateBoxPipeline(), "no_plate", None),
    ]:
        app = create_app(pipeline=pipeline, jobs_root=tmp_path / f"jobs-{expected_status}")
        client = TestClient(app)
        video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=3)

        submit = client.post("/api/infer/video", files={"file": ("clip.mp4", video_bytes, "video/mp4")})
        job_id = submit.json()["job_id"]

        job = None
        for _ in range(50):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.1)

        assert job is not None and job["status"] == "done", job and job.get("error_message")
        assert len(job["vehicle_sightings"]) == 1
        sighting = job["vehicle_sightings"][0]
        assert sighting["plate_status"] == expected_status
        assert sighting["plate_text"] is None
        assert sighting["raw_ocr_text"] == expected_raw

        thumb_response = client.get(sighting["thumbnail_url"])
        assert thumb_response.status_code == 200


@dataclass
class _ReacquiredVehiclePipeline:
    """Same physical vehicle, but the tracker sees it as two separate
    tracks (box jumps to a non-overlapping position partway through,
    simulating a brief tracking loss/re-acquisition) — both settle on the
    same plate text. The gallery must still show one card, not two (see
    docs/decisions.md #40's dedupe-by-plate-text)."""

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        box = (
            BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
            if frame_index < 2
            else BoundingBox(x_min=500, y_min=500, x_max=560, y_max=560)
        )
        return FrameResult(
            frame_index=frame_index,
            vehicles=[
                VehicleDetection(
                    box=box,
                    vehicle_type="car",
                    detection_confidence=0.9,
                    plate=PlateReading(
                        box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
                        raw_text="23AB123",
                        normalized_text="23 AB 123",
                        is_format_valid=True,
                        detection_confidence=0.9,
                        ocr_confidence=0.8,
                    ),
                )
            ],
        )


def test_infer_video_dedupes_two_tracks_that_settle_on_the_same_plate(tmp_path: Path) -> None:
    app = create_app(pipeline=_ReacquiredVehiclePipeline(), jobs_root=tmp_path / "jobs")
    client = TestClient(app)
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=4)

    submit = client.post("/api/infer/video", files={"file": ("clip.mp4", video_bytes, "video/mp4")})
    job_id = submit.json()["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)

    assert job is not None and job["status"] == "done", job and job.get("error_message")
    # 2 raw tracks (non-overlapping boxes), but same plate text -> 1 card.
    assert len(job["vehicle_sightings"]) == 1
    assert job["vehicle_sightings"][0]["plate_text"] == "23 AB 123"


@dataclass
class _TwoVehiclesPipeline:
    """Two distinct, fixed-position vehicles (different boxes -> different
    tracks) present in every frame — reproduces the real bug: before the
    fix, vehicle_type_counts summed per-frame detections (e.g. 2325 for 2
    real vehicles across 901 frames of arac3.mp4), not per distinct
    vehicle (see docs/decisions.md #39).
    """

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        return FrameResult(
            frame_index=frame_index,
            vehicles=[
                VehicleDetection(
                    box=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50),
                    vehicle_type="car",
                    detection_confidence=0.9,
                    plate=None,
                ),
                VehicleDetection(
                    box=BoundingBox(x_min=500, y_min=500, x_max=560, y_max=560),
                    vehicle_type="motorcycle",
                    detection_confidence=0.9,
                    plate=None,
                ),
            ],
        )


def test_infer_video_vehicle_type_counts_are_per_track_not_per_frame(tmp_path: Path) -> None:
    app = create_app(pipeline=_TwoVehiclesPipeline(), jobs_root=tmp_path / "jobs")
    client = TestClient(app)
    video_bytes = _tiny_mp4_bytes(tmp_path, frame_count=6)

    submit = client.post("/api/infer/video", files={"file": ("clip.mp4", video_bytes, "video/mp4")})
    job_id = submit.json()["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)

    assert job is not None and job["status"] == "done", job and job.get("error_message")
    # 2 real vehicles across 6 frames -> {"car": 1, "motorcycle": 1}, not
    # {"car": 6, "motorcycle": 6}.
    assert job["vehicle_type_counts"] == {"car": 1, "motorcycle": 1}


def test_job_status_404_for_unknown_id(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_camera_websocket_round_trip(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    with client.websocket_connect("/ws/camera") as ws:
        ws.send_bytes(_jpeg_bytes())
        payload = ws.receive_json()
        assert payload["vehicles"][0]["plate_text"] == "34 AB 123"
        annotated = ws.receive_bytes()
        assert len(annotated) > 0
