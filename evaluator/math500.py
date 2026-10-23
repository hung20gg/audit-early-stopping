import math
import warnings
import re
import signal
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional, Union

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
except ImportError:  # Symbolic checking becomes optional.
    sympy = None


Answer = Union[bool, int, float, str]


class EvaluationTimeout(TimeoutError):
    pass


class _SymbolicSyntaxWarning(SyntaxError):
    """A parser warning that invalidates semantic comparison."""


@contextmanager
def _time_limit(seconds: int, enabled: bool = True):
    """Best-effort timeout for SymPy on Unix main threads."""
    if not enabled or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise EvaluationTimeout("Symbolic comparison timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _extract_braced(text: str, opening_index: int) -> tuple[str, int]:
    """Extract a balanced {...} group whose opening brace is at opening_index."""
    if opening_index >= len(text) or text[opening_index] != "{":
        return "", opening_index

    depth = 0
    output: list[str] = []
    for index in range(opening_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth > 1:
                output.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(output), index + 1
            output.append(char)
        else:
            output.append(char)

    return "", opening_index


def _last_boxed_answer(text: str) -> Optional[str]:
    """Return the content of the last valid \boxed{...} or \fbox{...}."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*", text))
    for match in reversed(matches):
        index = match.end()
        while index < len(text) and text[index].isspace():
            index += 1

        if index < len(text) and text[index] == "{":
            answer, end_index = _extract_braced(text, index)
            if end_index > index and answer.strip():
                return answer.strip()
        else:
            # Supports uncommon output such as \boxed 42.
            tail = text[index:].strip()
            if tail:
                return re.split(r"[\n$]", tail, maxsplit=1)[0].strip(" .")
    return None


def _clean_extracted_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^[:=\s]+", "", answer)
    answer = re.sub(r"\s*(?:I hope.*|Therefore.*)$", "", answer, flags=re.I | re.S)
    answer = answer.strip()

    # Remove common display-math wrappers.
    wrappers = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if answer.startswith(left) and answer.endswith(right):
                answer = answer[len(left) : -len(right)].strip()
                changed = True

    return answer.rstrip(".。 \n\t")


def _clean_unboxed_answer(answer: str) -> str:
    """Clean an unboxed answer, preserving braces that denote a set."""
    answer = _clean_extracted_answer(answer)
    if (
        len(answer) >= 2
        and answer.startswith("{")
        and answer.endswith("}")
        and not any(separator in answer[1:-1] for separator in (",", ";", "|"))
    ):
        inner, end_index = _extract_braced(answer, 0)
        if end_index == len(answer):
            return inner.strip()
    return answer


def extract_final_answer(value: Answer) -> str:
    """
    Extract the final answer from a model response or a ground-truth string.

    Priority:
      1. Last \boxed{...} / \fbox{...}
      2. Last explicit final-answer phrase
      3. Entire input for a short answer
      4. Last display/inline math expression
      5. Last non-empty line
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).strip()
    if not text:
        return ""

    boxed = _last_boxed_answer(text)
    if boxed is not None:
        return _clean_extracted_answer(boxed)

    triggers = list(
        re.finditer(
            r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)|"
            r"therefore\s*,?\s*(?:the\s+answer\s+is)?|"
            r"thus\s*,?\s*(?:the\s+answer\s+is)?",
            text,
            flags=re.I,
        )
    )
    if triggers:
        tail = text[triggers[-1].end() :].strip()
        if tail:
            first_line = next((line.strip() for line in tail.splitlines() if line.strip()), tail)
            return _clean_unboxed_answer(first_line)

    # Ground truths such as "p - q" should remain intact.
    if "\n" not in text and len(text) <= 300:
        return _clean_unboxed_answer(text)

    display_math = re.findall(r"\\\[(.*?)\\\]", text, flags=re.S)
    if display_math:
        return _clean_unboxed_answer(display_math[-1])

    inline_math = re.findall(r"(?<!\\)\$(?!\$)(.*?)(?<!\\)\$", text, flags=re.S)
    if inline_math:
        return _clean_unboxed_answer(inline_math[-1])

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        return _clean_unboxed_answer(nonempty_lines[-1])

    return _clean_unboxed_answer(text)


def _strip_outer_group(text: str) -> str:
    pairs = {"(": ")", "[": "]", "{": "}"}
    while len(text) >= 2 and text[0] in pairs and text[-1] == pairs[text[0]]:
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == text[0]:
                depth += 1
            elif char == text[-1]:
                depth -= 1
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        text = text[1:-1]
    return text


def normalize_text(value: Answer) -> str:
    """Normalize harmless formatting differences without changing the mathematics."""
    text = extract_final_answer(value)
    text = text.strip().lower()

    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "×": r"\times",
        "÷": "/",
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\;": "",
        r"\:": "",
        r"\ ": "",
        r"\operatorname": "",
        r"\mathrm": "",
        r"\mathbf": "",
        r"\textstyle": "",
        r"\displaystyle": "",
        r"\%": "%",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Preserve the content of common text/font commands.
    text = re.sub(r"\\(?:text|rm|bf|it)\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。,")

    # Normalize a simple variable assignment such as x=2 to 2.
    assignment = re.fullmatch(r"[a-z][a-z0-9_]*=(.+)", text)
    if assignment:
        text = assignment.group(1)

    return _strip_outer_group(text)


def _replace_latex_fractions(text: str) -> str:
    """Convert balanced \frac{a}{b} expressions into ((a)/(b))."""
    output: list[str] = []
    index = 0
    while index < len(text):
        frac_match = re.match(r"\\frac\s*", text[index:])
        if not frac_match:
            output.append(text[index])
            index += 1
            continue

        cursor = index + frac_match.end()
        if cursor >= len(text) or text[cursor] != "{":
            output.append(text[index])
            index += 1
            continue

        numerator, after_num = _extract_braced(text, cursor)
        if after_num <= cursor:
            output.append(text[index])
            index += 1
            continue

        cursor = after_num
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            output.append(text[index])
            index += 1
            continue

        denominator, after_den = _extract_braced(text, cursor)
        if after_den <= cursor:
            output.append(text[index])
            index += 1
            continue

        numerator = _replace_latex_fractions(numerator)
        denominator = _replace_latex_fractions(denominator)
        output.append(f"(({numerator})/({denominator}))")
        index = after_den

    return "".join(output)


def _to_sympy_text(value: Answer) -> str:
    text = normalize_text(value)
    text = _replace_latex_fractions(text)

    # Convert common LaTeX into syntax accepted by SymPy's parser.
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\sqrt\(([^()]*)\)", r"sqrt(\1)", text)
    text = text.replace(r"\pi", "pi")
    text = text.replace(r"\infty", "oo")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    return text


def parse_digits(value: Answer, timeout: bool = True) -> Optional[float]:
    """Parse a purely numeric answer, including fractions and simple LaTeX."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = normalize_text(value)
    if not text:
        return None

    text = text.replace(",", "")
    text = text.rstrip("%")  # Percentage equivalence is handled in math_equal.
    text = _replace_latex_fractions(text)
    text = text.replace(r"\pi", "pi")
    text = text.replace("^", "**")

    # Fast exact path for ordinary integers, decimals, and a/b fractions.
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return float(int(text))
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", text, re.I):
            return float(Decimal(text))
        if re.fullmatch(r"[+-]?\d+/[+-]?\d+", text):
            return float(Fraction(text))
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None

    if sympy is None:
        return None

    try:
        with _time_limit(1, enabled=timeout):
            expr = _parse_symbolic_text(text)
            if expr is not None and not expr.free_symbols and expr.is_number:
                number = float(sympy.N(expr, 30))
                return number if math.isfinite(number) else None
    except Exception:
        return None
    return None


def numeric_equal(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _parse_symbolic_text(text: str):
    if sympy is None:
        return None

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    local_dict = {
        "pi": sympy.pi,
        "e": sympy.E,
        "oo": sympy.oo,
        "sqrt": sympy.sqrt,
    }
    # SymPy compiles parser input internally.  Malformed model answers such as
    # f^{-1}(0) can be interpreted as a set literal followed by a call.
    # Preserve this condition so those samples are never textually coalesced.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        try:
            result = parse_expr(
                text,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            )
        except Exception as error:
            if caught:
                raise _SymbolicSyntaxWarning(str(caught[0].message)) from error
            raise
    if caught:
        raise _SymbolicSyntaxWarning(str(caught[0].message))
    return result


def _has_symbolic_syntax_warning(text: str, timeout: bool = True) -> bool:
    try:
        with _time_limit(1, enabled=timeout):
            _parse_symbolic_text(text)
    except _SymbolicSyntaxWarning:
        return True
    except Exception:
        return False
    return False


def symbolic_equal(prediction: Answer, reference: Answer, timeout: bool = True) -> bool:
    if sympy is None:
        return False

    pred_text = _to_sympy_text(prediction)
    ref_text = _to_sympy_text(reference)
    if not pred_text or not ref_text:
        return False

    try:
        with _time_limit(3, enabled=timeout):
            pred_expr = _parse_symbolic_text(pred_text)
            ref_expr = _parse_symbolic_text(ref_text)
            if pred_expr is None or ref_expr is None:
                return False
            return bool(sympy.simplify(pred_expr - ref_expr) == 0)
    except (Exception, EvaluationTimeout):
        return False


def math_equal(
    prediction: Answer,
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
) -> bool:
    pred_answer = extract_final_answer(prediction)
    ref_answer = extract_final_answer(reference)

    pred_number = parse_digits(pred_answer)
    ref_number = parse_digits(ref_answer)
    if pred_number is not None and ref_number is not None:
        references = [ref_number]
        if include_percentage:
            references.extend([ref_number / 100.0, ref_number * 100.0])

        if is_close:
            return any(numeric_equal(pred_number, item) for item in references)
        return pred_number in references

    # A SyntaxWarning marks malformed input; do not let subsequent normalization
    # turn it into a matching expression.  Each such continuation stays unique.
    if _has_symbolic_syntax_warning(pred_answer, timeout=timeout) or _has_symbolic_syntax_warning(ref_answer, timeout=timeout):
        return False
    return symbolic_equal(pred_answer, ref_answer, timeout=timeout)
