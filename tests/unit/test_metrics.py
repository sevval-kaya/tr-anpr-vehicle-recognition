import pytest

from plaka.evaluation.metrics import (
    character_error_rate,
    exact_match_rate,
    top_k_accuracy,
)


class TestCharacterErrorRate:
    def test_perfect_predictions_yield_zero_cer(self) -> None:
        assert character_error_rate(["34AB123"], ["34AB123"]) == 0.0

    def test_single_substitution(self) -> None:
        # 1 edit out of 7 reference characters.
        assert character_error_rate(["34AB124"], ["34AB123"]) == pytest.approx(1 / 7)

    def test_aggregates_across_samples(self) -> None:
        predictions = ["34AB123", "06CD4567"]
        references = ["34AB123", "06CD456"]
        # 0 edits + 1 insertion, over 7 + 7 = 14 reference chars.
        assert character_error_rate(predictions, references) == pytest.approx(1 / 14)

    def test_empty_input_is_zero(self) -> None:
        assert character_error_rate([], []) == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            character_error_rate(["34AB123"], [])


class TestExactMatchRate:
    def test_all_correct(self) -> None:
        assert exact_match_rate(["34AB123", "06CD456"], ["34AB123", "06CD456"]) == 1.0

    def test_partial_correct(self) -> None:
        assert exact_match_rate(["34AB123", "WRONG"], ["34AB123", "06CD456"]) == 0.5

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            exact_match_rate([], [])


class TestTopKAccuracy:
    def test_top_1_hit(self) -> None:
        predictions = [["renault_clio", "fiat_egea"]]
        assert top_k_accuracy(predictions, ["renault_clio"], k=1) == 1.0

    def test_top_1_miss_but_top_2_hit(self) -> None:
        predictions = [["fiat_egea", "renault_clio"]]
        assert top_k_accuracy(predictions, ["renault_clio"], k=1) == 0.0
        assert top_k_accuracy(predictions, ["renault_clio"], k=2) == 1.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            top_k_accuracy([["a"]], ["a"], k=0)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            top_k_accuracy([], [], k=1)
