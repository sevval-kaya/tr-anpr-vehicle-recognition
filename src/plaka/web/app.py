"""FastAPI web front-end for InferencePipeline: upload a photo, upload a
video, or stream the browser's own camera — all three reuse the exact
same pipeline/annotation code as scripts/run_inference.py and
scripts/run_inference_video.py (src/plaka/pipeline/builder.py +
visualization.py), nothing is reimplemented for the web.

Run it with:

    python scripts/run_web.py

`create_app()` builds a real InferencePipeline by default (heavy: loads
the detector/OCR models once at startup, not per request). Tests pass in
a fake pipeline instead — see tests/unit/test_web_app.py — so the web
layer itself can be exercised without torch/ultralytics/paddleocr.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from plaka.pipeline.builder import REPO_ROOT, build_pipeline_from_config
from plaka.pipeline.inference_pipeline import InferencePipeline
from plaka.pipeline.schemas import FrameResult
from plaka.pipeline.visualization import annotate_frame
from plaka.utils.logging import get_logger
from plaka.web.jobs import JobManager

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"
DEFAULT_JOBS_ROOT = REPO_ROOT / "outputs" / "web_jobs"

# Live-camera frames are re-encoded JPEGs from a <canvas>; keep quality
# moderate — this runs once per frame, over a WebSocket, ideally several
# times a second.
CAMERA_JPEG_QUALITY = 80


def _decode_upload(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode uploaded image data")
    return image


def _encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buffer)


def _infer_and_annotate(
    pipeline: InferencePipeline,
    lock: threading.Lock,
    image: np.ndarray,
    jpeg_quality: int,
) -> tuple[FrameResult, bytes]:
    """Runs the actual model inference + drawing + JPEG encode — the
    genuinely CPU/GPU-bound, blocking part of a request. Called via
    `asyncio.to_thread` from every route below so it never runs on the
    asyncio event loop thread itself: FastAPI route handlers are `async
    def`, and a synchronous multi-model inference call (tens to hundreds
    of ms) made directly inside one blocks that single event loop for
    everyone else on it — every other in-flight HTTP/WebSocket connection
    stalls until it returns. That's what made the live-camera view "kasılıp
    donuyor" (stutter/freeze): each frame's inference was blocking the same
    loop the WebSocket itself runs on. `lock` serializes access to the
    shared pipeline/model objects across the thread pool used by
    asyncio.to_thread and JobManager's own background thread — model
    objects (ultralytics/paddleocr) aren't documented as safe for
    concurrent inference calls from multiple threads.
    """
    with lock:
        result = pipeline.process_frame(image)
    annotated = annotate_frame(image, result)
    return result, _encode_jpeg(annotated, quality=jpeg_quality)


def _summarize_vehicles(result: FrameResult) -> list[dict[str, Any]]:
    summary = []
    for vehicle in result.vehicles:
        plate = vehicle.plate
        summary.append(
            {
                "vehicle_type": vehicle.vehicle_type,
                "detection_confidence": round(vehicle.detection_confidence, 3),
                "plate_text": (plate.normalized_text or plate.raw_text) if plate else None,
                "plate_valid": bool(plate.is_format_valid) if plate else False,
                "plate_confidence": round(plate.ocr_confidence, 3) if plate else None,
            }
        )
    return summary


def create_app(
    pipeline: InferencePipeline | None = None,
    jobs_root: Path | None = None,
    pipeline_config_path: Path = DEFAULT_PIPELINE_CONFIG,
) -> FastAPI:
    if pipeline is None:
        logger.info("building InferencePipeline from %s ...", pipeline_config_path)
        pipeline = build_pipeline_from_config(pipeline_config_path)
        logger.info("pipeline ready")

    jobs_root = jobs_root or DEFAULT_JOBS_ROOT
    jobs_root.mkdir(parents=True, exist_ok=True)
    inference_lock = threading.Lock()
    job_manager = JobManager(pipeline=pipeline, jobs_root=jobs_root, inference_lock=inference_lock)

    app = FastAPI(title="Plaka Tanıma Sistemi")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.post("/api/infer/image")
    async def infer_image(file: UploadFile) -> JSONResponse:
        image = _decode_upload(await file.read())
        result, annotated_jpeg = await asyncio.to_thread(
            _infer_and_annotate, pipeline, inference_lock, image, 90
        )
        annotated_b64 = base64.b64encode(annotated_jpeg).decode("ascii")
        return JSONResponse(
            {
                "vehicles": _summarize_vehicles(result),
                "annotated_image_base64": annotated_b64,
            }
        )

    @app.post("/api/infer/video")
    async def infer_video(
        file: UploadFile,
        rotate: int = Form(0),
        sample_interval_seconds: float | None = Form(None),
    ) -> JSONResponse:
        if rotate not in (0, 90, 180, 270):
            raise HTTPException(status_code=400, detail="rotate must be one of 0, 90, 180, 270")
        if sample_interval_seconds is not None and sample_interval_seconds <= 0:
            raise HTTPException(status_code=400, detail="sample_interval_seconds must be > 0")
        video_bytes = await file.read()
        if not video_bytes:
            raise HTTPException(status_code=400, detail="empty upload")
        job_id = job_manager.submit(
            video_bytes,
            file.filename or "upload.mp4",
            rotate_degrees=rotate,
            sample_interval_seconds=sample_interval_seconds,
        )
        return JSONResponse({"job_id": job_id})

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return JSONResponse(
            {
                "job_id": job.job_id,
                "status": job.status,
                "progress": round(job.progress, 4),
                "frames_processed": job.frames_processed,
                "total_frames": job.total_frames,
                "frame_stride": job.frame_stride,
                "estimated_seconds_remaining": (
                    round(job.estimated_seconds_remaining, 1)
                    if job.estimated_seconds_remaining is not None
                    else None
                ),
                "error_message": job.error_message,
                "vehicle_type_counts": job.vehicle_type_counts,
                "vehicle_sightings": [
                    {
                        "frame_index": s.frame_index,
                        "timestamp_seconds": round(s.timestamp_seconds, 2),
                        "vehicle_type": s.vehicle_type,
                        "plate_status": s.plate_status,
                        "plate_text": s.plate_text,
                        "raw_ocr_text": s.raw_ocr_text,
                        "observation_count": s.observation_count,
                        "thumbnail_url": s.thumbnail_url,
                    }
                    for s in job.vehicle_sightings
                ],
                "video_url": f"/api/jobs/{job.job_id}/video" if job.output_video_path else None,
            }
        )

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str) -> FileResponse:
        job = job_manager.get(job_id)
        if job is None or job.output_video_path is None or not job.output_video_path.exists():
            raise HTTPException(status_code=404, detail="video not ready")
        return FileResponse(
            job.output_video_path, media_type="video/mp4", filename="plaka_annotated.mp4"
        )

    @app.get("/api/jobs/{job_id}/thumbnail/{name}")
    def job_thumbnail(job_id: str, name: str) -> FileResponse:
        job = job_manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        # `name` is generated by JobManager itself (plate_NN.jpg), but guard
        # against path traversal from the URL regardless.
        thumb_path = (job.output_dir / name).resolve()
        if job.output_dir.resolve() not in thumb_path.parents or not thumb_path.exists():
            raise HTTPException(status_code=404, detail="unknown thumbnail")
        return FileResponse(thumb_path, media_type="image/jpeg")

    @app.websocket("/ws/camera")
    async def camera_feed(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                # Offloaded to a worker thread (see _infer_and_annotate) so
                # one slow frame never blocks this event loop — and with it,
                # every other connection the server is holding open.
                result, annotated_jpeg = await asyncio.to_thread(
                    _infer_and_annotate, pipeline, inference_lock, image, CAMERA_JPEG_QUALITY
                )
                await websocket.send_json({"vehicles": _summarize_vehicles(result)})
                await websocket.send_bytes(annotated_jpeg)
        except WebSocketDisconnect:
            logger.info("camera websocket disconnected")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def create_app_from_env() -> FastAPI:
    """uvicorn factory entrypoint (`plaka.web.app:create_app_from_env`,
    factory=True) — uvicorn's factory mode only accepts a bare import
    string with no arguments, so scripts/run_web.py passes the
    --pipeline-config path through an env var instead of a function call.
    """
    import os

    config_path = Path(os.environ.get("PLAKA_PIPELINE_CONFIG", str(DEFAULT_PIPELINE_CONFIG)))
    return create_app(pipeline_config_path=config_path)
