"""Wires up a real (non-fake) InferencePipeline from the on-disk configs
(configs/pipeline.yaml + detection.yaml/ocr.yaml/classification.yaml).

Shared by every script that needs the actual pipeline running against real
images or video (scripts/run_inference.py, scripts/run_inference_video.py)
so the wiring is written once and stays consistent between them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plaka.classification.vehicle_classifier import VehicleClassifier
from plaka.detection.plate_detector import PlateDetector
from plaka.detection.vehicle_detector import VehicleDetector
from plaka.ocr.plate_ocr import PlateOcr
from plaka.pipeline.inference_pipeline import InferencePipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def build_pipeline_from_config(pipeline_config_path: Path) -> InferencePipeline:
    """Instantiate every real pipeline stage from the on-disk configs.

    Paths inside each sub-config are resolved relative to the repo root
    (matching how the config files themselves write them, e.g.
    "models/plate_detector/best.pt"), except the vehicle detector's
    default COCO checkpoint name, which ultralytics resolves/downloads
    itself and must be left as-is.
    """
    pipeline_config = _load_yaml(pipeline_config_path)
    config_dir = pipeline_config_path.parent

    detection = _load_yaml(config_dir / Path(pipeline_config["detection_config"]).name)
    ocr = _load_yaml(config_dir / Path(pipeline_config["ocr_config"]).name)

    vehicle_detector = VehicleDetector(
        weights_path=detection["vehicle_detector"]["weights_path"],
        confidence_threshold=detection["vehicle_detector"]["confidence_threshold"],
    )
    plate_detector = PlateDetector(
        weights_path=REPO_ROOT / detection["plate_detector"]["weights_path"],
        confidence_threshold=detection["plate_detector"]["confidence_threshold"],
    )
    plate_ocr = PlateOcr(weights_path=ocr["plate_ocr"]["weights_path"])

    # Make/model classification is opt-in — see configs/pipeline.yaml
    # "classification.enabled" and docs/decisions.md #29. Disabled by
    # default, but the wiring stays here so it's a one-line config flip
    # (not a code change) to bring back once labeled volume justifies it.
    vehicle_classifier: VehicleClassifier | None = None
    classification_enabled = bool(pipeline_config.get("classification", {}).get("enabled", False))
    if classification_enabled:
        classification = _load_yaml(
            config_dir / Path(pipeline_config["classification_config"]).name
        )
        vehicle_classifier = VehicleClassifier(
            weights_path=REPO_ROOT / classification["vehicle_classifier"]["weights_path"],
            class_names_path=REPO_ROOT / classification["vehicle_classifier"]["class_names_path"],
            architecture=classification["vehicle_classifier"]["architecture"],
            device=classification["vehicle_classifier"]["device"],
        )

    return InferencePipeline(
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        plate_ocr=plate_ocr,
        vehicle_classifier=vehicle_classifier,
    )
