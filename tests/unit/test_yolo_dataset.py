from pathlib import Path

import pytest
import yaml

from plaka.data.yolo_dataset import (
    YoloExample,
    find_yolo_examples,
    materialize_split,
    split_examples,
)


def _make_flat_source(root: Path, n: int, with_labels: bool = True) -> Path:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    for i in range(n):
        (images_dir / f"img{i}.jpg").write_bytes(b"fake-image-bytes")
        if with_labels:
            (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


class TestFindYoloExamples:
    def test_finds_flat_layout(self, tmp_path: Path) -> None:
        source = _make_flat_source(tmp_path / "flat", n=3)
        examples = find_yolo_examples(source)
        assert len(examples) == 3
        assert all(e.label_path.exists() for e in examples)

    def test_finds_roboflow_style_per_split_layout(self, tmp_path: Path) -> None:
        root = tmp_path / "roboflow"
        _make_flat_source(root / "train", n=2)
        _make_flat_source(root / "valid", n=1)
        _make_flat_source(root / "test", n=1)

        examples = find_yolo_examples(root)
        assert len(examples) == 4

    def test_missing_label_file_is_still_a_valid_example(self, tmp_path: Path) -> None:
        source = _make_flat_source(tmp_path / "flat", n=2, with_labels=False)
        examples = find_yolo_examples(source)
        assert len(examples) == 2
        assert not any(e.label_path.exists() for e in examples)

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        source = _make_flat_source(tmp_path / "flat", n=1)
        (source / "images" / "readme.txt").write_text("not an image")
        examples = find_yolo_examples(source)
        assert len(examples) == 1

    def test_missing_source_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_yolo_examples(tmp_path / "does_not_exist")


class TestSplitExamples:
    def _examples(self, n: int) -> list[YoloExample]:
        return [
            YoloExample(image_path=Path(f"img{i}.jpg"), label_path=Path(f"img{i}.txt"))
            for i in range(n)
        ]

    def test_split_sizes_respect_ratios(self) -> None:
        result = split_examples(self._examples(100), train_ratio=0.8, val_ratio=0.1, seed=42)
        assert len(result["train"]) == 80
        assert len(result["val"]) == 10
        assert len(result["test"]) == 10

    def test_split_is_deterministic_given_seed(self) -> None:
        examples = self._examples(50)
        first = split_examples(examples, seed=7)
        second = split_examples(examples, seed=7)
        assert [e.image_path for e in first["train"]] == [e.image_path for e in second["train"]]

    def test_split_covers_every_example_exactly_once(self) -> None:
        examples = self._examples(37)
        result = split_examples(examples, train_ratio=0.7, val_ratio=0.2, seed=1)
        all_paths = (
            [e.image_path for e in result["train"]]
            + [e.image_path for e in result["val"]]
            + [e.image_path for e in result["test"]]
        )
        assert sorted(all_paths) == sorted(e.image_path for e in examples)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            split_examples([])

    def test_invalid_ratios_raise(self) -> None:
        with pytest.raises(ValueError):
            split_examples(self._examples(10), train_ratio=0.9, val_ratio=0.2)


class TestMaterializeSplit:
    def test_writes_images_labels_and_data_yaml(self, tmp_path: Path) -> None:
        source = _make_flat_source(tmp_path / "source", n=4)
        examples = find_yolo_examples(source)
        split = split_examples(examples, train_ratio=0.5, val_ratio=0.25, seed=0)

        output_dir = tmp_path / "processed"
        data_yaml_path = materialize_split(split, output_dir, class_names=["license_plate"])

        assert data_yaml_path == output_dir / "data.yaml"
        assert (output_dir / "train" / "images").exists()
        assert (output_dir / "train" / "labels").exists()
        assert len(list((output_dir / "train" / "images").iterdir())) == 2
        assert len(list((output_dir / "val" / "images").iterdir())) == 1
        assert len(list((output_dir / "test" / "images").iterdir())) == 1

        config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        assert config["nc"] == 1
        assert config["names"] == ["license_plate"]
        assert config["train"] == "train/images"

    def test_missing_labels_get_empty_placeholder(self, tmp_path: Path) -> None:
        source = _make_flat_source(tmp_path / "source", n=2, with_labels=False)
        examples = find_yolo_examples(source)
        split = {"train": examples, "val": [], "test": []}

        output_dir = tmp_path / "processed"
        materialize_split(split, output_dir, class_names=["license_plate"])

        label_files = list((output_dir / "train" / "labels").iterdir())
        assert len(label_files) == 2
        assert all(f.read_text() == "" for f in label_files)
