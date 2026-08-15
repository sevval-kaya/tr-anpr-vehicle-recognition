import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_classifier_dataset import (
    build_class_name,
    materialize_dataset,
    normalize_class_component,
    resolve_class_directory_name,
)


class TestNormalizeClassComponent:
    def test_lowercases_and_strips(self) -> None:
        assert normalize_class_component("  Renault  ") == "renault"

    def test_collapses_spaces_and_hyphens_to_underscore(self) -> None:
        assert normalize_class_component("Mercedes Benz") == "mercedes_benz"
        assert normalize_class_component("E-Class") == "e_class"

    def test_drops_invalid_characters(self) -> None:
        assert normalize_class_component("Citroën C4") == "citron_c4"


class TestBuildClassName:
    def test_combines_make_and_model(self) -> None:
        assert build_class_name("Ford", "Mustang") == "ford_mustang"

    def test_blank_make_returns_none(self) -> None:
        assert build_class_name("", "clio") is None

    def test_blank_model_returns_none(self) -> None:
        assert build_class_name("renault", "") is None

    def test_make_only_ignores_model(self) -> None:
        assert build_class_name("Renault", "Clio", make_only=True) == "renault"

    def test_make_only_accepts_blank_model(self) -> None:
        assert build_class_name("Renault", "", make_only=True) == "renault"

    def test_make_only_still_requires_make(self) -> None:
        assert build_class_name("", "clio", make_only=True) is None


class TestResolveClassDirectoryName:
    def test_reuses_existing_directory_case_insensitively(self, tmp_path: Path) -> None:
        (tmp_path / "Toyota").mkdir()
        assert resolve_class_directory_name("toyota", "toyota", tmp_path) == "Toyota"

    def test_reuses_all_caps_existing_directory(self, tmp_path: Path) -> None:
        (tmp_path / "FIAT").mkdir()
        assert resolve_class_directory_name("fiat", "Fiat", tmp_path) == "FIAT"

    def test_title_cases_a_genuinely_new_class(self, tmp_path: Path) -> None:
        (tmp_path / "Toyota").mkdir()
        assert resolve_class_directory_name("renault", "renault", tmp_path) == "Renault"

    def test_nonexistent_output_dir_falls_back_to_title_case(self, tmp_path: Path) -> None:
        assert resolve_class_directory_name("opel", "opel", tmp_path / "nope") == "Opel"


def _make_source_images(source_dir: Path, filenames: list[str]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (source_dir / name).write_bytes(b"fake-image-bytes")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_file", "make", "model", "visibility"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestMaterializeDatasetFreshMode:
    def test_materializes_imagefolder_layout(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg", "2.jpg", "3.jpg"])
        csv_path = tmp_path / "labels.csv"
        _write_csv(
            csv_path,
            [
                {"image_file": "1.jpg", "make": "Ford", "model": "Mustang", "visibility": "clear"},
                {"image_file": "2.jpg", "make": "Ford", "model": "Mustang", "visibility": "clear"},
                {"image_file": "3.jpg", "make": "Renault", "model": "Clio", "visibility": "partial"},
            ],
        )

        output_dir = tmp_path / "out"
        counts = materialize_dataset([csv_path], source_dir, output_dir, min_images_per_class=1)

        assert counts == {"ford_mustang": 2, "renault_clio": 1}
        assert (output_dir / "ford_mustang" / "1.jpg").exists()
        assert (output_dir / "ford_mustang" / "2.jpg").exists()
        assert (output_dir / "renault_clio" / "3.jpg").exists()

    def test_reads_multiple_csvs(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg", "2.jpg"])
        csv_a = tmp_path / "a.csv"
        csv_b = tmp_path / "b.csv"
        _write_csv(csv_a, [{"image_file": "1.jpg", "make": "toyota", "model": "corolla", "visibility": "clear"}])
        _write_csv(csv_b, [{"image_file": "2.jpg", "make": "toyota", "model": "corolla", "visibility": "clear"}])

        output_dir = tmp_path / "out"
        counts = materialize_dataset([csv_a, csv_b], source_dir, output_dir, min_images_per_class=1)

        assert counts == {"toyota_corolla": 2}

    def test_skips_not_visible_and_blank_rows(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg", "2.jpg"])
        csv_path = tmp_path / "labels.csv"
        _write_csv(
            csv_path,
            [
                {"image_file": "1.jpg", "make": "", "model": "", "visibility": "not_visible"},
                {"image_file": "2.jpg", "make": "toyota", "model": "corolla", "visibility": "clear"},
            ],
        )

        output_dir = tmp_path / "out"
        counts = materialize_dataset([csv_path], source_dir, output_dir, min_images_per_class=1)

        assert counts == {"toyota_corolla": 1}

    def test_drops_classes_below_min_images_per_class(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg", "2.jpg", "3.jpg"])
        csv_path = tmp_path / "labels.csv"
        _write_csv(
            csv_path,
            [
                {"image_file": "1.jpg", "make": "toyota", "model": "corolla", "visibility": "clear"},
                {"image_file": "2.jpg", "make": "toyota", "model": "corolla", "visibility": "clear"},
                {"image_file": "3.jpg", "make": "audi", "model": "a4", "visibility": "clear"},
            ],
        )

        output_dir = tmp_path / "out"
        counts = materialize_dataset([csv_path], source_dir, output_dir, min_images_per_class=2)

        assert counts == {"toyota_corolla": 2}
        assert not (output_dir / "audi_a4").exists()

    def test_missing_listed_file_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg"])
        csv_path = tmp_path / "labels.csv"
        _write_csv(
            csv_path,
            [
                {"image_file": "1.jpg", "make": "toyota", "model": "corolla", "visibility": "clear"},
                {"image_file": "missing.jpg", "make": "fiat", "model": "egea", "visibility": "clear"},
            ],
        )

        output_dir = tmp_path / "out"
        counts = materialize_dataset([csv_path], source_dir, output_dir, min_images_per_class=1)

        assert counts == {"toyota_corolla": 1}

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            materialize_dataset([tmp_path / "nope.csv"], source_dir, tmp_path / "out")

    def test_missing_source_dir_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [])
        with pytest.raises(FileNotFoundError):
            materialize_dataset([csv_path], tmp_path / "nope", tmp_path / "out")


class TestMaterializeDatasetMakeOnly:
    def test_groups_by_make_ignoring_model(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg", "2.jpg"])
        csv_path = tmp_path / "labels.csv"
        _write_csv(
            csv_path,
            [
                {"image_file": "1.jpg", "make": "renault", "model": "clio", "visibility": "clear"},
                {"image_file": "2.jpg", "make": "renault", "model": "", "visibility": "clear"},
            ],
        )

        output_dir = tmp_path / "out"
        counts = materialize_dataset(
            [csv_path], source_dir, output_dir, make_only=True, min_images_per_class=1
        )

        assert counts == {"renault": 2}


class TestMaterializeDatasetMergeMode:
    def test_never_wipes_existing_content(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg"])
        output_dir = tmp_path / "out"
        (output_dir / "Toyota").mkdir(parents=True)
        (output_dir / "Toyota" / "existing.jpg").write_bytes(b"pre-existing")
        (output_dir / "BMW").mkdir()
        (output_dir / "BMW" / "existing2.jpg").write_bytes(b"pre-existing")

        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [{"image_file": "1.jpg", "make": "toyota", "model": "", "visibility": "clear"}])

        materialize_dataset([csv_path], source_dir, output_dir, make_only=True, merge=True)

        assert (output_dir / "Toyota" / "existing.jpg").exists()
        assert (output_dir / "BMW" / "existing2.jpg").exists()
        assert (output_dir / "Toyota" / "1.jpg").exists()

    def test_new_make_gets_title_cased_directory(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg"])
        output_dir = tmp_path / "out"
        (output_dir / "Toyota").mkdir(parents=True)

        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [{"image_file": "1.jpg", "make": "renault", "model": "", "visibility": "clear"}])

        counts = materialize_dataset([csv_path], source_dir, output_dir, make_only=True, merge=True)

        assert counts == {"Renault": 1}
        assert (output_dir / "Renault" / "1.jpg").exists()

    def test_sparse_class_is_not_dropped_in_merge_mode(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg"])
        output_dir = tmp_path / "out"

        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [{"image_file": "1.jpg", "make": "peugeot", "model": "", "visibility": "clear"}])

        # min_images_per_class=2 would normally drop a 1-image class, but
        # merge mode must add everything the caller explicitly asked for.
        counts = materialize_dataset(
            [csv_path], source_dir, output_dir, make_only=True, merge=True, min_images_per_class=2
        )

        assert counts == {"Peugeot": 1}

    def test_rerun_is_idempotent_does_not_recopy_or_double_count(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg"])
        output_dir = tmp_path / "out"

        csv_path = tmp_path / "labels.csv"
        _write_csv(csv_path, [{"image_file": "1.jpg", "make": "opel", "model": "", "visibility": "clear"}])

        first = materialize_dataset([csv_path], source_dir, output_dir, make_only=True, merge=True)
        second = materialize_dataset([csv_path], source_dir, output_dir, make_only=True, merge=True)

        assert first == {"Opel": 1}
        assert second == {"Opel": 0}  # nothing new to add

    def test_dry_run_touches_nothing(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        _make_source_images(source_dir, ["1.jpg"])
        output_dir = tmp_path / "out"
        (output_dir / "Toyota").mkdir(parents=True)

        csv_path = tmp_path / "labels.csv"
        _write_csv(
            csv_path,
            [
                {"image_file": "1.jpg", "make": "renault", "model": "", "visibility": "clear"},
            ],
        )

        counts = materialize_dataset(
            [csv_path], source_dir, output_dir, make_only=True, merge=True, dry_run=True
        )

        assert counts == {"Renault": 1}
        assert not (output_dir / "Renault").exists()
