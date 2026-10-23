import re
from typing import Union


Answer = Union[bool, int, float, str]
BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]+)\}", flags=re.IGNORECASE)


def normalize_options(value: Answer, *, require_box: bool = False) -> list[str] | None:
    text = str(value).strip()
    boxed_matches = BOXED_PATTERN.findall(text)
    if boxed_matches:
        text = boxed_matches[-1]
    elif require_box:
        return None

    options = [option.strip().lower() for option in text.split(",")]
    if not options or any(not option for option in options):
        return None
    return options


def multiple_choice_equal(prediction: Answer, reference: Answer) -> bool:
    """Compare a boxed, comma-separated model answer with the reference options."""
    predicted_options = normalize_options(prediction, require_box=True)
    reference_options = normalize_options(reference)
    return predicted_options is not None and predicted_options == reference_options


def check_is_correct(prediction: Answer, reference: Answer) -> bool:
    return multiple_choice_equal(prediction, reference)
