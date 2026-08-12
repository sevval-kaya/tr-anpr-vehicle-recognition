from pathlib import Path

import pytest

from plaka.data.datasets import discover_class_names, write_class_names


def test_discover_class_names_sorts_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "renault_clio").mkdir()
    (tmp_path / "fiat_egea").mkdir()
    (tmp_path / "togg_t10x").mkdir()
    (tmp_path / "not_a_class.txt").write_text("ignored")

    assert discover_class_names(tmp_path) == ["fiat_egea", "renault_clio", "togg_t10x"]


def test_discover_class_names_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_class_names(tmp_path / "does_not_exist")


def test_discover_class_names_empty_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        discover_class_names(tmp_path)


def test_write_class_names_round_trips(tmp_path: Path) -> None:
    class_names = ["fiat_egea", "renault_clio", "togg_t10x"]
    output_path = tmp_path / "classes.txt"

    write_class_names(class_names, output_path)

    assert output_path.read_text(encoding="utf-8").splitlines() == class_names
