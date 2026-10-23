import re
from math import isclose
from typing import Union


ParsedAnswer = Union[float, str]

NUMBER_PATTERN = re.compile(
    r"[-+]?\d+\s*/\s*[-+]?\d+"
    r"|[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    r"|[-+]?\.\d+%?"
)


def normalize_text(value: Union[bool, float, int, str]) -> str:
    return str(value).strip().rstrip(".").strip().lower()


def extract_final_answer(
    value: Union[bool, float, int, str],
) -> ParsedAnswer:
    """
    Extract and fully parse the final answer.

    Returns:
        float: when the final answer is numeric, including fractions
               and percentages.
        str: normalized text when no numeric answer is found.
    """
    text = str(value).strip()

    # Prefer the content after the final GSM8K-style answer marker.
    if "####" in text:
        text = text.rsplit("####", 1)[-1].strip()

    # Prefer a LaTeX boxed answer.
    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match:
        text = boxed_match.group(1).strip()
    else:
        # Otherwise use the last numeric expression.
        number_matches = NUMBER_PATTERN.findall(text)
        if number_matches:
            text = number_matches[-1].strip()
        else:
            return normalize_text(text)

    # Parse the extracted answer.
    numeric_text = text.replace(",", "").strip()

    is_percentage = numeric_text.endswith("%")
    if is_percentage:
        numeric_text = numeric_text[:-1].strip()

    try:
        if "/" in numeric_text:
            numerator, denominator = numeric_text.split("/", 1)
            number = float(numerator.strip()) / float(denominator.strip())
        else:
            number = float(numeric_text)
    except (ValueError, ZeroDivisionError):
        return normalize_text(text)

    return number / 100 if is_percentage else number


def numeric_equal(
    prediction: float,
    reference: float,
    *,
    abs_tol: float = 1e-4,
) -> bool:
    return isclose(
        prediction,
        reference,
        rel_tol=0.0,
        abs_tol=abs_tol,
    )


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
) -> bool:
    # Only two extraction/parsing calls.
    pred_answer = extract_final_answer(prediction)
    ref_answer = extract_final_answer(reference)

    if isinstance(pred_answer, float) and isinstance(ref_answer, float):
        references = [ref_answer]

        if include_percentage:
            references.extend([
                ref_answer / 100,
                ref_answer * 100,
            ])

        if is_close:
            return any(
                numeric_equal(pred_answer, candidate)
                for candidate in references
            )

        return pred_answer in references

    return pred_answer == ref_answer


def check_is_correct(pred, gt, timeout=True):
    return math_equal(pred, gt, timeout=timeout)