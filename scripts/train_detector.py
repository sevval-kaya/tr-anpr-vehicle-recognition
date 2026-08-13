#!/usr/bin/env python
"""Train the plate detector (YOLO) on a prepared dataset such as
data/processed/plates/ (see scripts/prepare_plate_data.py).

    python scripts/train_detector.py data/processed/plates/data.yaml

Writes the best checkpoint to models/plate_detector/best.pt, matching the
path PlateDetector reads by default (configs/detection.yaml), and reports
mAP@0.5 / mAP@0.5:0.95 / precision / recall from a post-training validation
pass. Detection mAP is not reimplemented in plaka.evaluation — Ultralytics'
own validator is used directly (see docs/decisions.md #5).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO  # type: ignore[attr-defined]

from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "detection.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "plate_detector"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs" / "detect"


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def train(
    data_yaml: Path,
    output_dir: Path,
    runs_dir: Path,
    config: dict[str, Any],
    epochs_override: int | None,
    image_size_override: int | None,
    batch_override: int | None,
    device_override: str | None,
    workers_override: int | None,
) -> Path:
    """Train the plate detector and copy the best checkpoint to `output_dir`.

    Ultralytics' own run artifacts (plots, batch previews, results.csv) are
    written under `runs_dir`, kept separate from `output_dir` so the served
    model location only ever holds the final checkpoint (matching
    models/vehicle_classifier/'s layout).

    Returns:
        Path to the copied best.pt checkpoint.
    """
    plate_detector_config = config["plate_detector"]
    training_config = config["training"]

    base_weights = f"{plate_detector_config['architecture']}.pt"
    epochs = epochs_override or int(training_config["epochs"])
    image_size = image_size_override or int(training_config["image_size"])
    batch = batch_override or int(training_config["batch_size"])
    patience = int(training_config["patience"])
    device = device_override or str(training_config.get("device", "cpu"))
    workers = (
        workers_override if workers_override is not None else int(training_config.get("workers", 8))
    )

    logger.info(
        "Training plate detector: base=%s epochs=%d imgsz=%d batch=%d device=%s workers=%d",
        base_weights,
        epochs,
        image_size,
        batch,
        device,
        workers,
    )

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        patience=patience,
        device=device,
        workers=workers,
        project=str(runs_dir),
        name=output_dir.name,
        exist_ok=True,
    )

    assert model.trainer is not None, "trainer is set by YOLO.train() on success"
    best_checkpoint = model.trainer.best
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"training finished but no best checkpoint found at {best_checkpoint}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "best.pt"
    shutil.copy2(best_checkpoint, final_path)
    logger.info("Best checkpoint copied to %s", final_path)

    metrics = model.val(data=str(data_yaml), imgsz=image_size, device=device)
    logger.info(
        "Validation: mAP50=%.4f mAP50-95=%.4f precision=%.4f recall=%.4f",
        metrics.box.map50,
        metrics.box.map,
        metrics.box.mp,
        metrics.box.mr,
    )

    return final_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "data_yaml",
        type=Path,
        help="Path to a YOLO data.yaml (e.g. data/processed/plates/data.yaml)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Where Ultralytics writes run artifacts (plots, previews, results.csv).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override configs/detection.yaml epochs."
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="Override configs/detection.yaml image_size."
    )
    parser.add_argument(
        "--batch", type=int, default=None, help="Override configs/detection.yaml batch_size."
    )
    parser.add_argument(
        "--device", type=str, default=None, help='e.g. "0" for the first GPU, "cpu" to force CPU.'
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Override configs/detection.yaml workers."
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    train(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        runs_dir=args.runs_dir,
        config=config,
        epochs_override=args.epochs,
        image_size_override=args.image_size,
        batch_override=args.batch,
        device_override=args.device,
        workers_override=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
