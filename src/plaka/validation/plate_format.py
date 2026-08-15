"""Validation of OCR output against the Turkish vehicle plate format.

Turkish plates follow: <il kodu 01-81> <harf grubu> <rakam grubu>, where the
letter/digit group lengths are constrained to a fixed set of combinations and
the letter set excludes characters that don't exist on Turkish plates
(Q, W, X, and the dotted/diacritic Turkish letters).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Turkish plates never use Q, W, X, or diacritic letters (Ç, Ğ, İ, Ö, Ş, Ü).
ALLOWED_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"

# (letter_count -> allowed digit_count set), per Turkish traffic regulation.
# 3-letter groups allow both 2 and 3 digits (e.g. "66 AAP 914") — the
# original 3: {2}-only table was disproven by real labeled plate photos
# during the OCR pilot (see docs/decisions.md).
VALID_GROUP_COMBINATIONS: dict[int, frozenset[int]] = {
    1: frozenset({4}),
    2: frozenset({3, 4}),
    3: frozenset({2, 3}),
}

_PLATE_PATTERN = re.compile(
    rf"^(?P<il_kodu>\d{{2}})\s*(?P<harf>[{ALLOWED_LETTERS}]{{1,3}})\s*(?P<rakam>\d{{2,4}})$"
)

MIN_IL_KODU = 1
MAX_IL_KODU = 81


@dataclass(frozen=True, slots=True)
class PlateValidationResult:
    """Outcome of validating a candidate plate string."""

    is_valid: bool
    normalized: str | None
    il_kodu: int | None = None
    harf_grubu: str | None = None
    rakam_grubu: str | None = None
    reason: str | None = None


class TurkishPlateValidator:
    """Validates and normalizes OCR output against the Turkish plate format.

    Used as the post-OCR filter described in the pipeline (stage 5): raw
    OCR text is normalized (whitespace/case/common confusions) and checked
    against il kodu range and harf/rakam group-length rules before being
    accepted as a final read.
    """

    def normalize(self, raw_text: str) -> str:
        """Uppercase, strip, and collapse internal whitespace of raw OCR text."""
        text = raw_text.strip().upper()
        text = re.sub(r"[^\dA-Z]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def validate(self, raw_text: str) -> PlateValidationResult:
        """Validate a raw OCR string, returning the normalized plate if valid."""
        normalized = self.normalize(raw_text)
        compact = normalized.replace(" ", "")

        match = _PLATE_PATTERN.match(compact)
        if match is None:
            match = _PLATE_PATTERN.match(normalized)

        if match is None:
            return PlateValidationResult(
                is_valid=False,
                normalized=normalized,
                reason="format_mismatch",
            )

        il_kodu = int(match.group("il_kodu"))
        harf_grubu = match.group("harf")
        rakam_grubu = match.group("rakam")

        if not (MIN_IL_KODU <= il_kodu <= MAX_IL_KODU):
            return PlateValidationResult(
                is_valid=False,
                normalized=normalized,
                il_kodu=il_kodu,
                harf_grubu=harf_grubu,
                rakam_grubu=rakam_grubu,
                reason="il_kodu_out_of_range",
            )

        allowed_digit_counts = VALID_GROUP_COMBINATIONS.get(len(harf_grubu))
        if allowed_digit_counts is None or len(rakam_grubu) not in allowed_digit_counts:
            return PlateValidationResult(
                is_valid=False,
                normalized=normalized,
                il_kodu=il_kodu,
                harf_grubu=harf_grubu,
                rakam_grubu=rakam_grubu,
                reason="invalid_group_combination",
            )

        formatted = f"{il_kodu:02d} {harf_grubu} {rakam_grubu}"
        return PlateValidationResult(
            is_valid=True,
            normalized=formatted,
            il_kodu=il_kodu,
            harf_grubu=harf_grubu,
            rakam_grubu=rakam_grubu,
        )
