"""Evaluation metrics for the plate OCR and make/model classification modules.

Detection metrics (mAP@0.5, mAP@0.5:0.95) are intentionally not reimplemented
here: Ultralytics ships a validated COCO-style mAP implementation
(`model.val()`), and re-deriving IoU-matching/mAP by hand would duplicate
well-tested code for no accuracy benefit. See docs/decisions.md.
"""

from __future__ import annotations

from collections.abc import Sequence


def _levenshtein_distance(source: str, target: str) -> int:
    """Compute character-level edit distance between two strings."""
    if source == target:
        return 0
    if len(source) == 0:
        return len(target)
    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current_row = [i] + [0] * len(target)
        for j, target_char in enumerate(target, start=1):
            insertion_cost = current_row[j - 1] + 1
            deletion_cost = previous_row[j] + 1
            substitution_cost = previous_row[j - 1] + (source_char != target_char)
            current_row[j] = min(insertion_cost, deletion_cost, substitution_cost)
        previous_row = current_row

    return previous_row[-1]


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Aggregate Character Error Rate: total edit distance / total reference length.

    Args:
        predictions: OCR outputs, one per sample.
        references: Ground-truth plate strings, one per sample, same order.

    Returns:
        CER in [0, +inf); 0.0 means every prediction exactly matched its
        reference. Returns 0.0 for an empty input (nothing to score).

    Raises:
        ValueError: if predictions and references have different lengths.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    if not references:
        return 0.0

    total_edits = sum(
        _levenshtein_distance(pred, ref) for pred, ref in zip(predictions, references, strict=False)
    )
    total_reference_chars = sum(len(ref) for ref in references)
    if total_reference_chars == 0:
        return 0.0 if total_edits == 0 else float("inf")

    return total_edits / total_reference_chars


def exact_match_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of predictions that exactly equal their reference (end-to-end read accuracy).

    Raises:
        ValueError: if predictions and references have different lengths, or
            both are empty.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    if not references:
        raise ValueError("cannot compute exact_match_rate on empty input")

    matches = sum(pred == ref for pred, ref in zip(predictions, references, strict=False))
    return matches / len(references)


def top_k_accuracy(
    ranked_predictions: Sequence[Sequence[str]],
    references: Sequence[str],
    k: int,
) -> float:
    """Fraction of samples where the true label appears in the top-k predictions.

    Args:
        ranked_predictions: one ranked list of predicted class labels per
            sample, most confident first (e.g. from softmax top-k).
        references: true class label per sample, same order.
        k: how many top predictions to consider; must be positive.

    Raises:
        ValueError: if inputs have mismatched lengths, are empty, or k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(ranked_predictions) != len(references):
        raise ValueError(
            f"ranked_predictions ({len(ranked_predictions)}) and references "
            f"({len(references)}) must have the same length"
        )
    if not references:
        raise ValueError("cannot compute top_k_accuracy on empty input")

    hits = sum(
        reference in candidates[:k]
        for candidates, reference in zip(ranked_predictions, references, strict=False)
    )
    return hits / len(references)
