import pytest

from plaka.validation.plate_format import TurkishPlateValidator


@pytest.fixture
def validator() -> TurkishPlateValidator:
    return TurkishPlateValidator()


@pytest.mark.parametrize(
    "raw_text,expected",
    [
        ("34 A 1234", "34 A 1234"),
        ("34a1234", "34 A 1234"),
        ("06 AB 123", "06 AB 123"),
        ("81 ABC 12", "81 ABC 12"),
        ("66 AAP 914", "66 AAP 914"),
        ("  34   AB   1234  ", "34 AB 1234"),
    ],
)
def test_valid_plates_are_normalized(
    validator: TurkishPlateValidator, raw_text: str, expected: str
) -> None:
    result = validator.validate(raw_text)
    assert result.is_valid is True
    assert result.normalized == expected


@pytest.mark.parametrize(
    "raw_text,expected_reason",
    [
        ("00 A 1234", "il_kodu_out_of_range"),
        ("82 A 1234", "il_kodu_out_of_range"),
        ("34 ABCD 12", "format_mismatch"),
        ("34 ABC 1234", "invalid_group_combination"),
        ("34 A 123", "invalid_group_combination"),
        ("34 AB 12", "invalid_group_combination"),
        ("34 QW 1234", "format_mismatch"),
        ("not a plate", "format_mismatch"),
        ("", "format_mismatch"),
    ],
)
def test_invalid_plates_are_rejected(
    validator: TurkishPlateValidator, raw_text: str, expected_reason: str
) -> None:
    result = validator.validate(raw_text)
    assert result.is_valid is False
    assert result.reason == expected_reason


def test_il_kodu_and_groups_are_parsed(validator: TurkishPlateValidator) -> None:
    result = validator.validate("34 AB 1234")
    assert result.il_kodu == 34
    assert result.harf_grubu == "AB"
    assert result.rakam_grubu == "1234"
