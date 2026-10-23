"""Score a boxed MMLU-Pro option letter, including LaTeX text wrappers."""

import re


def extract_option(prediction: str) -> str | None:
    text = str(prediction)
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not boxes:
        return None
    start = boxes[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                value = text[start:index].strip()
                while True:
                    wrapped = re.fullmatch(r"\\(?:text|textit|textbf|mathrm|mathbf|rm|it|bf)\s*\{(.*)\}", value, re.S)
                    if not wrapped:
                        break
                    value = wrapped.group(1).strip()
                return value.upper() if re.fullmatch(r"[A-J]", value, re.I) else None
    return None


def multiple_choice_equal(prediction: str, reference: str) -> bool:
    expected = str(reference).strip().upper()
    return bool(re.fullmatch(r"[A-J]", expected)) and extract_option(prediction) == expected
