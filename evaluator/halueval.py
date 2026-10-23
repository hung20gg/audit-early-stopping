"""Answer evaluation for HaluEval's binary labels."""

from typing import Union

from evaluator.math500 import normalize_text


Answer = Union[bool, int, float, str]
VALID_LABELS = {"yes", "no"}


def label_equal(prediction: Answer, reference: Answer) -> bool:
    """Compare extracted yes/no labels from full or replayed answers."""
    predicted_label = normalize_text(prediction)
    reference_label = normalize_text(reference)
    return (
        predicted_label in VALID_LABELS
        and reference_label in VALID_LABELS
        and predicted_label == reference_label
    )


def check_is_correct(prediction: Answer, reference: Answer) -> bool:
    return label_equal(prediction, reference)
