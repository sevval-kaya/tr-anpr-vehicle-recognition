import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from download_weights import _release_asset_url, download_weights  # noqa: E402


class TestReleaseAssetUrl:
    def test_builds_the_expected_github_release_url(self) -> None:
        url = _release_asset_url("owner/repo", "v0.1.0", "best.pt")
        assert url == "https://github.com/owner/repo/releases/download/v0.1.0/best.pt"


class TestDownloadWeights:
    def test_skips_download_when_file_already_exists(self, tmp_path: Path) -> None:
        dest = tmp_path / "best.pt"
        dest.write_bytes(b"already here")
        calls = []

        download_weights(dest=dest, _download_fn=lambda url, path: calls.append((url, path)))

        assert calls == []  # never invoked
        assert dest.read_bytes() == b"already here"  # untouched

    def test_force_redownloads_even_if_present(self, tmp_path: Path) -> None:
        dest = tmp_path / "best.pt"
        dest.write_bytes(b"stale")

        def fake_download(url: str, path: Path) -> None:
            path.write_bytes(b"fresh")

        download_weights(dest=dest, force=True, _download_fn=fake_download)

        assert dest.read_bytes() == b"fresh"

    def test_downloads_when_missing(self, tmp_path: Path) -> None:
        dest = tmp_path / "nested" / "best.pt"  # parent dir doesn't exist yet
        calls = []

        def fake_download(url: str, path: Path) -> None:
            calls.append(url)
            path.write_bytes(b"weights")

        result = download_weights(
            dest=dest,
            repo="owner/repo",
            tag="v9.9.9",
            asset_name="x.pt",
            _download_fn=fake_download,
        )

        assert result == dest
        assert dest.read_bytes() == b"weights"
        assert calls == ["https://github.com/owner/repo/releases/download/v9.9.9/x.pt"]

    def test_http_error_raises_a_clear_system_exit(self, tmp_path: Path) -> None:
        dest = tmp_path / "best.pt"

        def failing_download(url: str, path: Path) -> None:
            raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

        with pytest.raises(SystemExit, match="404"):
            download_weights(dest=dest, _download_fn=failing_download)
        assert not dest.exists()

    def test_url_error_raises_a_clear_system_exit(self, tmp_path: Path) -> None:
        dest = tmp_path / "best.pt"

        def failing_download(url: str, path: Path) -> None:
            raise urllib.error.URLError("no route to host")

        with pytest.raises(SystemExit, match="internet connection"):
            download_weights(dest=dest, _download_fn=failing_download)
        assert not dest.exists()
