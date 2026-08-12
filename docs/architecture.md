# Architecture

Source of truth for scope/requirements: the project brief
("Türk Plaka + Araç Marka/Model Tanıma Sistemi", 12 Ağustos 2026). This file
maps that brief onto the actual repo structure and records where the code
deviates from or extends it.

## Pipeline stages → code

| Doc step | Module | Status |
|---|---|---|
| 2. Araç tespiti | `plaka.detection.vehicle_detector.VehicleDetector` | Implemented, wraps COCO-pretrained YOLO — no training needed for baseline |
| 3b. Plaka tespiti | `plaka.detection.plate_detector.PlateDetector` | Implemented, **requires a trained checkpoint** (no free pretrained equivalent) |
| 4. Ön işleme | `plaka.ocr.preprocessing.enhance_plate_crop` | Implemented: CLAHE contrast only (see decisions.md for why perspective correction is deferred) |
| 4. OCR | `plaka.ocr.plate_ocr.PlateOcr` | Implemented, wraps PaddleOCR, restricted to plate charset |
| 5. Format doğrulama | `plaka.validation.plate_format.TurkishPlateValidator` | Implemented and tested, no model dependency |
| 3a. Marka/model | `plaka.classification.vehicle_classifier.VehicleClassifier` | Implemented, wraps a timm backbone, **requires a trained checkpoint** |
| 6. Zamansal oylama | *(not yet built)* | Deferred — needs a tracker (e.g. ByteTrack) + video data; roadmap stage 3+ |
| 1-7 orchestration | `plaka.pipeline.inference_pipeline.InferencePipeline` | Implemented for single frames; plate↔vehicle association via `BoundingBox.containment_ratio` (see decisions.md) |
| 7. Çıktı şeması | `plaka.pipeline.schemas` | Implemented (pydantic models: `FrameResult`, `VehicleDetection`, `PlateReading`, `MakeModelPrediction`) |

`InferencePipeline` depends on `plaka.pipeline.protocols` (structural
interfaces), not on the concrete detector/OCR/classifier classes directly —
this keeps orchestration logic testable with lightweight fakes and keeps
each stage's backend swappable independently (e.g. replacing PaddleOCR with
a custom CRNN later touches only `PlateOcr`, not the pipeline).

## Repo layout

```
src/plaka/
  detection/       vehicle + plate detection (YOLO wrappers)
  ocr/              plate OCR + preprocessing
  classification/   make/model classification (timm wrapper)
  validation/       Turkish plate format validator (no ML)
  pipeline/         schemas, protocols, single-frame orchestration
  data/             dataset layout helpers (ImageFolder discovery)
  evaluation/       metrics: CER, exact-match, top-k accuracy
  utils/            logging
configs/            per-stage YAML configs (weights paths, thresholds, hyperparams)
scripts/            CLI entry points (dataset download, training, inference)
tests/unit/         one test module per src module, no external deps required
tests/integration/  reserved for tests that need real weights/data (empty for now)
data/               raw/ external/ processed/ — gitignored, see data/README.md
models/             trained checkpoints — gitignored, not yet populated
```

## Open design questions

See `docs/decisions.md` for decisions already made and their rationale, and
the plan summary delivered alongside this scaffold for decisions still
pending user sign-off (dataset scope for the first training run, target
classifier size/architecture, deployment target).
