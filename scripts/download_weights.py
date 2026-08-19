#!/usr/bin/env python
"""Download the trained plate-detector checkpoint from this repo's GitHub
Release, so a fresh clone can run the pipeline without training one from
scratch (see docs/decisions.md #44). Idempotent: skips the download if
the destination file already exists (`--force` to re-download anyway).

    python scripts/download_weights.py
    python scripts/download_weights.py --force
    python scripts/download_weights.py --tag v0.2.0

No GitHub authentication needed — release assets on a public repo are
served over plain HTTPS, same mechanism as scripts/download_datasets.py's
_download_with_progress.

NOTE: as of this script's addition, the v0.1.0 release referenced below
has NOT been created yet — see docs/decisions.md #44. Running this
script before the release exists will fail with a clear 404 message;
that's expected until the release is published.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from plaka.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "sevval-kaya/tr-anpr-vehicle-recognition"
DEFAULT_TAG = "v0.1.0"
DEFAULT_ASSET_NAME = "best.pt"
DEFAULT_DEST = REPO_ROOT / "models" / "plate_detector" / "best.pt"


def _release_asset_url(repo: str, tag: str, asset_name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"


def _download_with_progress(url: str, output_path: Path) -> None:
    """Downloads to a `.part` sibling file first, then renames into place
    — an interrupted download (network drop, Ctrl-C) never leaves a
    truncated file at `output_path` that later silently loads as a
    corrupt/incomplete checkpoint.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "plaka-weights-downloader"})
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with urllib.request.urlopen(request, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        with (
            tmp_path.open("wb") as out_file,
            tqdm(
                total=total_size or None, unit="B", unit_scale=True, unit_divisor=1024
            ) as progress,
        ):
            while chunk := response.read(1024 * 1024):
                out_file.write(chunk)
                progress.update(len(chunk))
    tmp_path.replace(output_path)


def download_weights(
    dest: Path = DEFAULT_DEST,
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
    asset_name: str = DEFAULT_ASSET_NAME,
    force: bool = False,
    _download_fn: Callable[[str, Path], None] = _download_with_progress,
) -> Path:
    """`_download_fn` is only overridden by tests (a fake that writes
    bytes locally instead of making a real HTTP request) — real callers
    always use the default, same dependency-injection pattern as
    plaka.web.app.create_app(pipeline=...).
    """
    if dest.exists() and not force:
        logger.info("%s already present, skipping download (use --force to re-download)", dest)
        return dest

    url = _release_asset_url(repo, tag, asset_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)

    try:
        _download_fn(url, dest)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"Weights download failed ({exc.code} {exc.reason}) for {url}\n"
            "Most likely: the release/tag/asset name doesn't exist yet, or you "
            "passed the wrong --repo/--tag/--asset-name. Check "
            f"https://github.com/{repo}/releases"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Weights download failed for {url}: {exc.reason}\n"
            "Check your internet connection and try again."
        ) from exc

    size_mb = dest.stat().st_size / 1e6
    logger.info("Done: %s (%.1f MB)", dest, size_mb)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/repo on GitHub")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag")
    parser.add_argument("--asset-name", default=DEFAULT_ASSET_NAME, help="Release asset filename")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination file already exists",
    )
    args = parser.parse_args()

    download_weights(
        dest=args.dest, repo=args.repo, tag=args.tag, asset_name=args.asset_name, force=args.force
    )


if __name__ == "__main__":
    main()
