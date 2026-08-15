from pathlib import Path

import pytest

from plaka.data.manual_labels import (
    load_label_csvs,
    merge_and_dedupe_labels,
    stratified_holdout_split,
)


def _rows(*entries: tuple[str, str, str, str, str]) -> list[tuple[str, dict[str, str]]]:
    """entries: (source, image_file, make, model, visibility)"""
    return [
        (source, {"image_file": f, "make": make, "model": model, "visibility": vis})
        for source, f, make, model, vis in entries
    ]


class TestLoadLabelCsvs:
    def test_reads_rows_from_every_csv_in_dir(self, tmp_path: Path) -> None:
        (tmp_path / "a.csv").write_text(
            "image_file,make,model,visibility\n1.jpg,toyota,corolla,clear\n", encoding="utf-8"
        )
        (tmp_path / "b.csv").write_text(
            "image_file,make,model,visibility\n2.jpg,fiat,egea,clear\n", encoding="utf-8"
        )

        rows = load_label_csvs(tmp_path)

        assert [r["image_file"] for _src, r in rows] == ["1.jpg", "2.jpg"]

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_label_csvs(tmp_path / "nope")


class TestMergeAndDedupeLabels:
    def test_basic_usable_rows_pass_through(self) -> None:
        rows = _rows(
            ("a.csv", "1.jpg", "toyota", "corolla", "clear"),
            ("a.csv", "2.jpg", "fiat", "egea", "clear"),
        )
        merged = merge_and_dedupe_labels(rows)
        assert merged.usable == {"1.jpg": ("toyota", "corolla"), "2.jpg": ("fiat", "egea")}
        assert merged.excluded_not_visible == []
        assert merged.conflicts == []

    def test_not_visible_and_blank_make_excluded(self) -> None:
        rows = _rows(
            ("a.csv", "1.jpg", "", "", "not_visible"),
            ("a.csv", "2.jpg", "toyota", "corolla", "clear"),
        )
        merged = merge_and_dedupe_labels(rows)
        assert merged.usable == {"2.jpg": ("toyota", "corolla")}
        assert merged.excluded_not_visible == ["1.jpg"]

    def test_consistent_duplicate_across_files_is_kept_once(self) -> None:
        rows = _rows(
            ("a.csv", "1.jpg", "volkswagen", "passat", "clear"),
            ("b.csv", "1.jpg", "volkswagen", "passat", "clear"),
        )
        merged = merge_and_dedupe_labels(rows)
        assert merged.usable == {"1.jpg": ("volkswagen", "passat")}

    def test_duplicate_prefers_the_entry_with_a_model(self) -> None:
        rows = _rows(
            ("a.csv", "1.jpg", "hyundai", "", "clear"),
            ("b.csv", "1.jpg", "hyundai", "accent", "clear"),
        )
        merged = merge_and_dedupe_labels(rows)
        assert merged.usable == {"1.jpg": ("hyundai", "accent")}

    def test_conflicting_make_across_files_is_reported_and_excluded(self) -> None:
        rows = _rows(
            ("a.csv", "1.jpg", "toyota", "corolla", "clear"),
            ("b.csv", "1.jpg", "honda", "civic", "clear"),
        )
        merged = merge_and_dedupe_labels(rows)
        assert "1.jpg" not in merged.usable
        assert len(merged.conflicts) == 1
        assert merged.conflicts[0].image_file == "1.jpg"
        assert set(merged.conflicts[0].entries) == {("a.csv", "toyota"), ("b.csv", "honda")}

    def test_make_is_lowercased_and_stripped(self) -> None:
        rows = _rows(("a.csv", "1.jpg", "  Toyota  ", " Corolla ", "clear"))
        merged = merge_and_dedupe_labels(rows)
        assert merged.usable == {"1.jpg": ("toyota", "corolla")}


class TestStratifiedHoldoutSplit:
    def test_holds_out_one_per_brand_with_at_least_two(self) -> None:
        usable = {
            "1.jpg": ("renault", "clio"),
            "2.jpg": ("renault", "megane"),
            "3.jpg": ("renault", "9"),
        }
        train, test, singles = stratified_holdout_split(usable, seed=42)
        assert len(test) == 1
        assert len(train) == 2
        assert set(train) | set(test) == set(usable)
        assert singles == []

    def test_single_image_brand_goes_entirely_to_train(self) -> None:
        usable = {"1.jpg": ("audi", "a3")}
        train, test, singles = stratified_holdout_split(usable, seed=42)
        assert train == {"1.jpg": ("audi", "a3")}
        assert test == {}
        assert singles == ["audi"]

    def test_mixed_brands(self) -> None:
        usable = {
            "1.jpg": ("renault", "clio"),
            "2.jpg": ("renault", "megane"),
            "3.jpg": ("audi", "a3"),
        }
        train, test, singles = stratified_holdout_split(usable, seed=42)
        assert len(test) == 1
        assert next(iter(test)).startswith(("1", "2"))  # the held-out one is a renault
        assert "3.jpg" in train  # audi (single) never held out
        assert singles == ["audi"]

    def test_deterministic_given_seed(self) -> None:
        usable = {f"{i}.jpg": ("renault", "clio") for i in range(10)}
        first = stratified_holdout_split(usable, seed=7)
        second = stratified_holdout_split(usable, seed=7)
        assert first == second

    def test_no_overlap_between_train_and_test(self) -> None:
        usable = {f"{i}.jpg": ("toyota", "corolla") for i in range(5)}
        train, test, _singles = stratified_holdout_split(usable, seed=1)
        assert set(train).isdisjoint(set(test))
