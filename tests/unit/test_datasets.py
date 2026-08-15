from pathlib import Path

import pytest

from plaka.data.datasets import (
    aggregate_images_by_brand,
    count_images_per_class,
    discover_class_names,
    extract_brand,
    select_target_classes,
    write_class_names,
)


def _make_class(root: Path, name: str, n_images: int) -> None:
    class_dir = root / name
    class_dir.mkdir(parents=True)
    for i in range(n_images):
        (class_dir / f"img{i}.jpg").write_bytes(b"fake")


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


class TestCountImagesPerClass:
    def test_counts_images_ignoring_other_files(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "fiat_egea", 3)
        (tmp_path / "fiat_egea" / "readme.txt").write_text("not an image")

        counts = count_images_per_class(tmp_path, ["fiat_egea"])

        assert counts == {"fiat_egea": 3}


class TestSelectTargetClasses:
    def test_balances_across_makes_round_robin(self, tmp_path: Path) -> None:
        # ford has many classes, renault has only one — a naive top-N by
        # count would drop renault entirely; round-robin must not.
        for i in range(10):
            _make_class(tmp_path, f"ford_focus_{2000 + i}", n_images=5)
        _make_class(tmp_path, "renault_clio_2015", n_images=5)

        selected = select_target_classes(
            tmp_path, target_makes=["ford", "renault"], max_classes=4
        )

        assert "renault_clio_2015" in selected
        assert len(selected) == 4

    def test_prefers_higher_image_count_within_a_make(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "fiat_egea_2018", n_images=2)
        _make_class(tmp_path, "fiat_egea_2019", n_images=20)

        selected = select_target_classes(tmp_path, target_makes=["fiat"], max_classes=1)

        assert selected == ["fiat_egea_2019"]

    def test_matches_space_separated_make_prefix(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "mercedes benz_190_1985", n_images=3)

        selected = select_target_classes(
            tmp_path, target_makes=["mercedes benz"], max_classes=5
        )

        assert selected == ["mercedes benz_190_1985"]

    def test_ignores_classes_outside_target_makes(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "fiat_egea_2018", n_images=3)
        _make_class(tmp_path, "acura_cl_1997", n_images=3)

        selected = select_target_classes(tmp_path, target_makes=["fiat"], max_classes=10)

        assert selected == ["fiat_egea_2018"]

    def test_no_matches_raises(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "acura_cl_1997", n_images=3)

        with pytest.raises(ValueError):
            select_target_classes(tmp_path, target_makes=["fiat"], max_classes=10)

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "ford_focus_2010", n_images=5)
        _make_class(tmp_path, "fiat_egea_2018", n_images=5)

        selected = select_target_classes(
            tmp_path, target_makes=["ford", "fiat"], max_classes=10
        )

        assert selected == sorted(selected)


class TestExtractBrand:
    def test_simple_make_model_year(self) -> None:
        assert extract_brand("renault_captur_2015") == "renault"

    def test_space_separated_make_has_no_internal_underscore(self) -> None:
        # "mercedes benz" contains a space, not an underscore, so splitting
        # on the first "_" only correctly leaves it intact as one token.
        assert extract_brand("mercedes benz_190_1985") == "mercedes benz"

    def test_multi_word_model(self) -> None:
        assert extract_brand("peugeot_407_wagon_2005") == "peugeot"

    def test_no_underscore_returns_whole_string(self) -> None:
        assert extract_brand("togg") == "togg"


class TestAggregateImagesByBrand:
    def test_pools_images_across_a_brands_raw_classes(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "toyota_corolla_2010", n_images=3)
        _make_class(tmp_path, "toyota_camry_2012", n_images=2)

        pools = aggregate_images_by_brand(
            tmp_path, target_brands=["toyota"], max_images_per_brand=100
        )

        assert len(pools["toyota"]) == 5

    def test_sparse_brand_with_one_raw_class_is_not_dropped(self, tmp_path: Path) -> None:
        for i in range(10):
            _make_class(tmp_path, f"ford_focus_{2000 + i}", n_images=50)
        _make_class(tmp_path, "renault_captur_2015", n_images=3)

        pools = aggregate_images_by_brand(
            tmp_path, target_brands=["ford", "renault"], max_images_per_brand=100
        )

        assert len(pools["renault"]) == 3
        assert len(pools["ford"]) == 100  # capped, would otherwise be 500

    def test_brand_below_cap_keeps_everything(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "opel_astra_2007", n_images=8)

        pools = aggregate_images_by_brand(
            tmp_path, target_brands=["opel"], max_images_per_brand=300
        )

        assert len(pools["opel"]) == 8

    def test_brand_not_in_dataset_is_omitted_not_an_error(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "toyota_corolla_2010", n_images=3)

        pools = aggregate_images_by_brand(
            tmp_path, target_brands=["toyota", "togg"], max_images_per_brand=100
        )

        assert "togg" not in pools
        assert "toyota" in pools

    def test_brands_outside_target_list_are_ignored(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "acura_cl_1997", n_images=5)
        _make_class(tmp_path, "toyota_corolla_2010", n_images=3)

        pools = aggregate_images_by_brand(
            tmp_path, target_brands=["toyota"], max_images_per_brand=100
        )

        assert set(pools) == {"toyota"}

    def test_deterministic_given_seed(self, tmp_path: Path) -> None:
        _make_class(tmp_path, "toyota_corolla_2010", n_images=50)

        first = aggregate_images_by_brand(
            tmp_path, target_brands=["toyota"], max_images_per_brand=10, seed=7
        )
        second = aggregate_images_by_brand(
            tmp_path, target_brands=["toyota"], max_images_per_brand=10, seed=7
        )

        assert first["toyota"] == second["toyota"]

    def test_missing_dataset_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            aggregate_images_by_brand(
                tmp_path / "nope", target_brands=["toyota"], max_images_per_brand=100
            )
