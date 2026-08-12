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
