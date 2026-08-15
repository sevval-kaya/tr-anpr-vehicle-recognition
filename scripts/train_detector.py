#!/usr/bin/env python
"""Train the plate detector (YOLO) on a prepared dataset such as
data/processed/plates/ (see scripts/prepare_plate_data.py).

    python scripts/train_detector.py data/processed/plates/data.yaml

    # Light fine-tune from an existing checkpoint instead of training from
    # scratch (small LR, frozen backbone, a handful of epochs):
    python scripts/train_detector.py data/processed/plate_finetune_arac3/data_finetune.yaml \
        --base-weights models/plate_detector/best.pt --lr0 0.0005 --freeze 10 \
        --epochs 15 --output-dir models/plate_detector_arac3_finetune

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
    cache_override: str | bool | None,
    base_weights_override: str | None = None,
    lr0_override: float | None = None,
    freeze_override: int | None = None,
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

    base_weights = base_weights_override or f"{plate_detector_config['architecture']}.pt"
    epochs = epochs_override or int(training_config["epochs"])
    image_size = image_size_override or int(training_config["image_size"])
    batch = batch_override or int(training_config["batch_size"])
    patience = int(training_config["patience"])
    device = device_override or str(training_config.get("device", "cpu"))
    workers = (
        workers_override if workers_override is not None else int(training_config.get("workers", 8))
    )
    # "ram" pre-decodes/resizes every image once and keeps it in memory,
    # instead of re-reading and re-decoding from disk every epoch — a real
    # win when source images are large (some of our plate photos are
    # 4608x2592) and there's headroom to spare (this dataset comfortably
    # fits: 5.4K images at 640x640x3 uint8 is a few GB).
    cache = cache_override if cache_override is not None else training_config.get("cache", False)

    logger.info(
        "Training plate detector: base=%s epochs=%d imgsz=%d batch=%d "
        "device=%s workers=%d cache=%s lr0=%s freeze=%s",
        base_weights,
        epochs,
        image_size,
        batch,
        device,
        workers,
        cache,
        lr0_override if lr0_override is not None else "(default)",
        freeze_override if freeze_override is not None else "(none)",
    )

    train_kwargs: dict[str, Any] = {}
    if lr0_override is not None:
        # Also lowers the final LR proportionally (lrf is a *fraction* of
        # lr0 in Ultralytics) so a light fine-tune doesn't end training on
        # BASELINE_lr0*default_lrf, which would be larger than lr0 itself.
        train_kwargs["lr0"] = lr0_override
    if freeze_override is not None:
        # Freezes the first N layers (backbone) so the fine-tune only
        # adapts the later/head layers to the new angle instead of
        # re-learning general plate-vs-background features from 19 images.
        train_kwargs["freeze"] = freeze_override

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        patience=patience,
        device=device,
        workers=workers,
        cache=cache,
        project=str(runs_dir),
        name=output_dir.name,
        exist_ok=True,
        **train_kwargs,
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
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=["ram", "disk", "false"],
        help="Image caching mode (speeds up epochs after the first, at the cost of memory/disk).",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default=None,
        help="Start from these weights instead of the fresh COCO-pretrained "
        "'{architecture}.pt' — e.g. models/plate_detector/best.pt for a light "
        "fine-tune rather than training from scratch.",
    )
    parser.add_argument(
        "--lr0", type=float, default=None, help="Override the initial learning rate."
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N layers (backbone) — keeps a fine-tune from a real "
        "checkpoint from re-learning general features off a tiny dataset.",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    cache_override: str | bool | None = args.cache
    if args.cache == "false":
        cache_override = False
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
        cache_override=cache_override,
        base_weights_override=args.base_weights,
        lr0_override=args.lr0,
        freeze_override=args.freeze,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
