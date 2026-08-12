# data/

Nothing under this directory is committed to git except this file and
`.gitkeep` placeholders (see `.gitignore`) — datasets are large and
license-encumbered, so they're fetched locally instead.

- `raw/` — untouched downloads (Kaggle exports, git clones, camera footage
  dumps). Never edited in place.
- `external/` — same idea, reserved for the specific open-source datasets
  `scripts/download_datasets.py` fetches (VMMRdb, Stanford Cars, Turkish
  plate dataset). Kept separate from `raw/` so third-party data and our own
  collected data can't get mixed up by accident.
- `processed/` — derived artifacts: resized crops, ImageFolder-formatted
  class directories, train/val/test splits. Reproducible from `raw/` +
  `external/` via scripts in `scripts/`, so it's safe to delete and
  regenerate.

See `docs/architecture.md` for how each dataset feeds into which model, and
`docs/decisions.md` for licensing notes per dataset.

## `external/` kaynak envanteri

| Klasör | Kaynak | Amaç | Lisans |
|---|---|---|---|
| `vmmrdb/` | faezetta/VMMRdb (Dropbox zip) | Marka/model sınıflandırma (baseline) | Bkz. üst repo; ticari kullanım öncesi doğrulanmalı |
| `roboflow_plates/` | [toggai/turkish-license-plate](https://universe.roboflow.com/toggai/turkish-license-plate) (Roboflow Universe) | Plaka **tespiti** — YOLO formatında bbox etiketli, tek sınıf (`license_plate`) | **CC BY 4.0** — dahili kullanım serbest; harici paylaşım/yayın durumunda "toggai, Turkish License Plate Dataset, Roboflow Universe" şeklinde atıf gerekir |
| `kaggle_plates/` | [smaildurcan/turkish-license-plate-dataset](https://www.kaggle.com/datasets/smaildurcan/turkish-license-plate-dataset) (Kaggle) | Plaka **tespiti** — sayfa metadata'sına göre YOLO formatında bbox etiketli (YOLOv5 için hazırlanmış), ~2.9GB. İndirilip doğrulanmadı, henüz `data/external/`e alınmadı. | **CC0: Public Domain** — atıf bile gerekmez |

Roboflow ve Kaggle indirmeleri kimlik doğrulama gerektirir (`ROBOFLOW_API_KEY`
ortam değişkeni / `~/.kaggle/kaggle.json`) — ikisi de Claude'un kendi
adına oluşturamayacağı, kullanıcıya ait hesap bilgileridir.
