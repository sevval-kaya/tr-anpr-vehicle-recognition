#!/usr/bin/env python
"""One-off diagnostic: is a video's "vehicles detected but no plates read"
symptom caused by the wrong --rotate value, or a genuine camera-distance/
resolution floor (docs/decisions.md #32/#33)? Runs the plate detector at
a low confidence threshold (0.05, well below production's 0.5) against
every one of the four possible rotations and reports which one produces
the most/largest vehicle-contained plate boxes.

    python scripts/diagnose_rotation.py path/to/clip.mp4
    python scripts/diagnose_rotation.py path/to/clip.mp4 --max-frames 200

The real orientation should show a clear, large jump over the other
three (a plate is a small, orientation-sensitive object — a 90°-off crop
essentially never contains a plate-shaped detection). If all four are
similarly weak/absent even at this permissive threshold, that points to
a genuine distance/resolution problem instead — rotation can't fix that
(see arac2.mp4, docs/decisions.md #32).

Manual, one-time tool — not wired into scripts/run_web.py's video job
pipeline, which needs a single user-chosen --rotate value per upload,
not a 4x-slower diagnostic sweep on every video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from plaka.detection.plate_detector import PlateDetector
from plaka.detection.vehicle_detector import VehicleDetector
from plaka.pipeline.builder import REPO_ROOT
from plaka.pipeline.video_io import VALID_ROTATIONS, apply_rotation
from plaka.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PLATE_WEIGHTS = REPO_ROOT / "models" / "plate_detector" / "best.pt"
DIAGNOSTIC_CONFIDENCE_THRESHOLD = 0.05
CONTAINMENT_THRESHOLD = 0.5

# Below this, even the *winning* rotation's boxes are in the same range as
# the confirmed-unreadable arac2.mp4 case (median ~32px, docs/decisions.md
# #32/#33) — a genuine resolution floor, not something --rotate can fix.
SMALL_BOX_WARNING_PX = 50.0


def _load_frames(video_path: Path, max_frames: int | None) -> list[NDArray[np.uint8]]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        read_ok, frame = capture.read()
        if not read_ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    return frames


def diagnose(video_path: Path, plate_weights: Path, max_frames: int | None) -> None:
    frames = _load_frames(video_path, max_frames)
    if not frames:
        raise SystemExit(f"could not read any frames from {video_path}")
    print(
        f"{video_path}: {len(frames)} frame(s) loaded "
        f"({'all' if max_frames is None else f'capped at --max-frames {max_frames}'})"
    )

    vehicle_detector = VehicleDetector(weights_path="yolo11n.pt", confidence_threshold=0.4)
    plate_detector = PlateDetector(
        weights_path=plate_weights, confidence_threshold=DIAGNOSTIC_CONFIDENCE_THRESHOLD
    )

    results: dict[int, dict[str, float]] = {}
    for rotation in sorted(VALID_ROTATIONS):
        frames_with_vehicle = 0
        frames_with_plate_match = 0
        matched_box_widths: list[float] = []
        for frame in frames:
            rotated = apply_rotation(frame, rotation)
            vehicles = vehicle_detector.detect(rotated)
            if vehicles:
                frames_with_vehicle += 1
            plates = plate_detector.detect(rotated)
            best_width_this_frame: float | None = None
            for vehicle in vehicles:
                for plate in plates:
                    if vehicle.box.containment_ratio(plate.box) >= CONTAINMENT_THRESHOLD and (
                        best_width_this_frame is None or plate.box.width > best_width_this_frame
                    ):
                        best_width_this_frame = plate.box.width
            if best_width_this_frame is not None:
                frames_with_plate_match += 1
                matched_box_widths.append(best_width_this_frame)

        matched_box_widths.sort()
        results[rotation] = {
            "frames_with_vehicle": frames_with_vehicle,
            "frames_with_plate_match": frames_with_plate_match,
            "median_box_width": matched_box_widths[len(matched_box_widths) // 2] if matched_box_widths else 0.0,
            "max_box_width": matched_box_widths[-1] if matched_box_widths else 0.0,
        }

    header = f"{'rotate':>8} | {'araçlı kare':>12} | {'plaka eşleşen kare':>19} | {'medyan kutu (px)':>17} | {'max kutu (px)':>14}"
    print(f"\n{header}")
    print("-" * len(header))
    for rotation, r in results.items():
        print(
            f"{rotation:>8} | {r['frames_with_vehicle']:>12.0f} | {r['frames_with_plate_match']:>19.0f} | "
            f"{r['median_box_width']:>17.1f} | {r['max_box_width']:>14.1f}"
        )

    best_rotation = max(results, key=lambda r: results[r]["frames_with_plate_match"])
    best = results[best_rotation]
    others_best_match = max(
        r["frames_with_plate_match"] for rot, r in results.items() if rot != best_rotation
    )

    print()
    if best["frames_with_plate_match"] == 0:
        print(
            "SONUÇ: Hiçbir döndürmede (0.05 eşikte bile) araç-içi bir plaka kutusu bulunamadı. "
            "Bu, yanlış döndürmeden değil muhtemelen gerçek bir kamera-mesafesi/çözünürlük "
            "sorunundan kaynaklanıyor (bkz. docs/decisions.md #32/#33, arac2.mp4 örneği) — "
            "döndürmeyi değiştirmek muhtemelen yardımcı olmayacaktır."
        )
    elif best["frames_with_plate_match"] > 2 * max(1, others_best_match):
        print(
            f"SONUÇ: rotate={best_rotation} açık ara en güçlü sinyali veriyor "
            f"({best['frames_with_plate_match']} kare vs diğerlerinde en fazla {others_best_match:.0f}) "
            f"— muhtemelen doğru döndürme buydu, önceki denemede başka bir değer kullanılmış olabilir."
        )
        if best["median_box_width"] < SMALL_BOX_WARNING_PX:
            print(
                f"UYARI: rotate={best_rotation}'de bile eşleşen kutular küçük "
                f"(medyan {best['median_box_width']:.1f}px, en büyük {best['max_box_width']:.1f}px) — "
                f"bu, arac2.mp4'teki doğrulanmış çözünürlük tabanıyla (medyan ~32px, hiçbir "
                f"eşikte/döndürmede okunamıyor, bkz. docs/decisions.md #32/#33) aynı büyüklük "
                f"mertebesinde. Doğru döndürmeyi bulmuş olsanız bile plakalar OCR için fiziksel "
                f"olarak çok küçük olabilir — bu, kamera mesafesi/zoom kaynaklı ayrı bir sorun."
            )
    else:
        print(
            f"SONUÇ: Sonuçlar döndürmeler arasında belirgin şekilde ayrışmıyor "
            f"(en iyisi rotate={best_rotation}: {best['frames_with_plate_match']} kare, "
            f"diğerlerinde en fazla {others_best_match:.0f}) — büyük ihtimalle döndürme sorunu "
            f"değil, kamera mesafesi/açısı kaynaklı gerçek bir zorluk."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path, help="Path to the video file to diagnose")
    parser.add_argument("--plate-weights", type=Path, default=DEFAULT_PLATE_WEIGHTS)
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Sample only the first N frames (faster)."
    )
    args = parser.parse_args()
    diagnose(args.video, args.plate_weights, args.max_frames)


if __name__ == "__main__":
    main()
