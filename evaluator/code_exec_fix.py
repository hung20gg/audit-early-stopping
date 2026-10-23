import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# Default per-problem execution timeout in seconds.
DEFAULT_TIMEOUT = 1


def _get_resource_limits_header() -> str:
    """Generate Python header to limit CPU affinity and memory usage."""
    return textwrap.dedent("""\
        import sys
        # 1. Limit each submission to one core, while distributing concurrent
        # submissions over all CPUs available to this process.
        try:
            import os
            import psutil
            p = psutil.Process()
            available_cpus = p.cpu_affinity()
            if available_cpus:
                selected_cpu = available_cpus[os.getpid() % len(available_cpus)]
                p.cpu_affinity([selected_cpu])
        except Exception:
            try:
                import os
                if hasattr(os, "sched_setaffinity"):
                    available_cpus = sorted(os.sched_getaffinity(0))
                    if available_cpus:
                        selected_cpu = available_cpus[
                            os.getpid() % len(available_cpus)
                        ]
                        os.sched_setaffinity(0, {selected_cpu})
            except Exception:
                pass

        # 2. Limit memory to 512 MB
        try:
            import resource
            mem_bytes = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass

        # 3. Memory limit polling/watchdog thread for Windows
        try:
            import threading
            import time
            import os
            def memory_limit_watchdog():
                try:
                    import psutil
                    proc = psutil.Process()
                    mem_limit = 512 * 1024 * 1024
                    while True:
                        if proc.memory_info().rss > mem_limit:
                            os._exit(1)
                        time.sleep(0.1)
                except Exception:
                    pass
            t = threading.Thread(target=memory_limit_watchdog, daemon=True)
            t.start()
        except Exception:
            pass
    """)




def extract_code_block(text: str) -> str:
    """Extract the last fenced Python code block from model output.

    If the model wraps its answer in ````python ... ``` `` we return the
    content.  Otherwise we strip the ``<think>...</think>`` block and return
    everything after the close tag (the "answer" portion).
    """
    # Try fenced code blocks first (```python ... ``` or ``` ... ```)
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()

    # Fall back: strip <think>...</think> and return the rest.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()
    return cleaned



def _build_humaneval_script(code: str, reference: dict, test_type: str = "full") -> str:
    """Combine generated code with HumanEval test harness."""
    limits_header = _get_resource_limits_header()

    if test_type == "public":
        public_tests = reference.get("public_tests", [])
        test_code = "\n".join(
            f"assert {case['input']} == {case['output']}" for case in public_tests
        )
        return f"{limits_header}\n\n{code}\n\n{test_code}\n"

    test_code = reference["test"]
    entry_point = reference["entry_point"]
    return (
        f"{limits_header}\n\n"
        f"{code}\n\n"
        f"{test_code}\n\n"
        f"check({entry_point})\n"
    )


def _parse_test_cases(raw: str) -> list:
    """Parse test cases from a JSON string or return an empty list."""
    if not raw:
        return []
    try:
        cases = json.loads(raw)
        if isinstance(cases, list):
            return cases
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        import base64
        import zlib
        import pickle
        decoded = base64.b64decode(raw.encode("utf-8"))
        decompressed = zlib.decompress(decoded)
        pickled = pickle.loads(decompressed)
        if isinstance(pickled, str):
            cases = json.loads(pickled)
        else:
            cases = pickled
        if isinstance(cases, list):
            return cases
    except Exception:
        pass

    return []


def _parse_metadata(reference: dict) -> dict:
    """Parse LiveCodeBench metadata.

    Some dataset versions use `metadata`, while others use `meta_data`.
    """
    raw = reference.get("metadata", reference.get("meta_data", {}))

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def _build_livecodebench_functional_script(
    code: str,
    reference: dict,
    all_cases: list,
) -> str:
    """Build a runner accepting either function calls or stdin/stdout.

    LiveCodeBench marks LeetCode-style cases as ``functional``.  Some model
    prompts nevertheless produce a complete Codeforces-style program.  Try
    both interfaces and accept the submission when either one passes every
    selected test case.

    The function interface checks ``Solution.<func_name>``, the named module
    function, and every other module-level function.  A candidate function
    must pass all cases; nested/helper calls are not independently scanned.
    """
    limits_header = _get_resource_limits_header()
    metadata = _parse_metadata(reference)
    func_name = metadata.get("func_name")

    if not func_name:
        raise ValueError("Functional test is missing metadata.func_name")

    cases_json = json.dumps(all_cases, ensure_ascii=False)

    runner_template = textwrap.dedent("""\
        import ast
        import io
        import json
        import sys

        _solution_code = {code!r}
        _test_cases = json.loads({cases_json!r})
        _func_name = {func_name!r}

        class _LimitedStringIO(io.StringIO):
            def __init__(self, max_chars):
                super().__init__()
                self._max_chars = max_chars
                self._written = 0

            def write(self, value):
                self._written += len(value)
                if self._written > self._max_chars:
                    raise OSError("submission output limit exceeded")
                return super().write(value)

        def _parse_arguments(raw):
            # The input is normally a Python argument expression:
            # '"0011"'       -> ["0011"]
            # '[1, 2], 3'    -> [[1, 2], 3]
            # Some LiveCodeBench releases put one argument on each line:
            # '[1, 2]\\n3'   -> [[1, 2], 3]
            if not isinstance(raw, str):
                return [raw]

            raw = raw.strip()
            if not raw:
                return []

            # Parse newline-separated expressions before wrapping the input
            # in a tuple.  Otherwise Python silently concatenates adjacent
            # string literals: '"a1"\\n"c3"' would become one "a1c3" value.
            if "\\n" in raw:
                try:
                    module = ast.parse(raw, mode="exec")
                    if len(module.body) > 1 and all(
                        isinstance(statement, ast.Expr)
                        for statement in module.body
                    ):
                        return [
                            ast.literal_eval(statement.value)
                            for statement in module.body
                        ]
                except (ValueError, SyntaxError):
                    pass

            try:
                return list(ast.literal_eval(f"({{raw}},)"))
            except (ValueError, SyntaxError):
                pass

            # Parse newline-separated Python literals as separate expression
            # statements.  Using the AST rather than splitlines also supports
            # a list/dict literal formatted across several lines.
            try:
                module = ast.parse(raw, mode="exec")
                if module.body and all(
                    isinstance(statement, ast.Expr)
                    for statement in module.body
                ):
                    return [
                        ast.literal_eval(statement.value)
                        for statement in module.body
                    ]
            except (ValueError, SyntaxError):
                pass

            # Fallback for newline-separated JSON-only values such as true,
            # false, and null.
            return [json.loads(line) for line in raw.splitlines() if line.strip()]

        def _parse_expected(raw):
            if not isinstance(raw, str):
                return raw

            raw = raw.strip()

            try:
                return ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return raw

        def _outputs_equal(actual, expected):
            actual = actual.strip()
            expected = expected.strip()

            # Prefer structured comparison so formatting differences such as
            # ``[[3,7]]`` versus ``[[3, 7]]`` do not matter.
            try:
                if _parse_expected(actual) == _parse_expected(expected):
                    return True
            except Exception:
                pass

            # Normal competitive-programming output comparison ignores
            # differences in whitespace between tokens.
            return actual.split() == expected.split()

        def _case_expected(case):
            return case.get("output", case.get("expected_output", "None"))

        def _passes_function(target_factory):
            for case in _test_cases:
                try:
                    target = target_factory()
                    if not callable(target):
                        return False
                    actual = target(*_parse_arguments(case.get("input", "")))
                    expected = _parse_expected(_case_expected(case))
                    if actual != expected:
                        return False
                except BaseException:
                    return False
            return True

        _namespace = {{"__name__": "submission"}}
        try:
            # ``submission`` prevents a guarded main() from running.  If an
            # unguarded program reads stdin here, retain any definitions made
            # before that exception and still try the stdin/stdout path below.
            exec(_solution_code, _namespace)
        except BaseException:
            pass

        _candidate_factories = []
        _solution_class = _namespace.get("Solution")
        if isinstance(_solution_class, type):
            _candidate_factories.append(
                lambda: getattr(_solution_class(), _func_name, None)
            )

        # AST discovery deliberately considers module-level functions only.
        # Put the metadata-named function first, then try every other outer
        # function.  The OR is between candidates, while each candidate must
        # satisfy every case.
        try:
            _tree = ast.parse(_solution_code)
            _outer_names = [
                node.name for node in _tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name != "main"
            ]
        except SyntaxError:
            _outer_names = []

        _ordered_names = [_func_name] + _outer_names
        _seen_names = set()
        for _name in _ordered_names:
            if _name in _seen_names:
                continue
            _seen_names.add(_name)
            _candidate = _namespace.get(_name)
            if callable(_candidate):
                _candidate_factories.append(
                    lambda candidate=_candidate: candidate
                )

        if any(_passes_function(factory) for factory in _candidate_factories):
            sys.exit(0)

        # No function candidate passed: retry the complete submission as a
        # normal stdin/stdout program, once in a fresh namespace per case.
        _stdio_passed = True
        for _case in _test_cases:
            _old_stdin, _old_stdout = sys.stdin, sys.stdout
            _captured = _LimitedStringIO({output_limit})
            sys.stdin = io.StringIO(str(_case.get("input", "")))
            sys.stdout = _captured
            try:
                exec(_solution_code, {{"__name__": "__main__"}})
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    _stdio_passed = False
            except BaseException:
                _stdio_passed = False
            finally:
                sys.stdin, sys.stdout = _old_stdin, _old_stdout

            if not _stdio_passed or not _outputs_equal(
                _captured.getvalue(), str(_case_expected(_case))
            ):
                _stdio_passed = False
                break

        if not _stdio_passed:
            print(
                "FAIL: neither an outer function nor stdin/stdout passed all cases",
                file=sys.stderr,
            )
            sys.exit(1)
    """)

    formatted_script = runner_template.format(
        code=code,
        cases_json=cases_json,
        func_name=func_name,
        output_limit=1024 * 1024,
    )

    return f"{limits_header}\n\n{formatted_script}"

def _build_livecodebench_script(code: str, reference: dict, test_type: str = "full") -> str:
    """Build an execution script for LiveCodeBench stdin/stdout problems."""
    public_cases = _parse_test_cases(reference.get("public_test_cases", "[]"))
    private_cases = _parse_test_cases(reference.get("private_test_cases", "[]"))
    if test_type == "public":
        all_cases = public_cases
    elif test_type == "private":
        all_cases = private_cases
    else:
        all_cases = public_cases + private_cases
    limits_header = _get_resource_limits_header()

    is_functional = any(
        str(case.get("testtype", "")).lower() == "functional"
        for case in all_cases
    )

    if is_functional:
        return _build_livecodebench_functional_script(
            code=code,
            reference=reference,
            all_cases=all_cases,
        )

    if not all_cases:
        # If there are no test cases, just check that the code is syntactically
        # valid by compiling it.
        return f"{limits_header}\n\nimport ast\nast.parse({code!r})\n"

    # LiveCodeBench problems use stdin/stdout.  We simulate this by patching
    # sys.stdin and capturing sys.stdout for each test case.
    runner_template = textwrap.dedent("""\
        import sys
        import io
        import json

        class _LimitedStringIO(io.StringIO):
            def __init__(self, max_chars):
                super().__init__()
                self._max_chars = max_chars
                self._written = 0
                self.limit_exceeded = False

            def write(self, value):
                self._written += len(value)
                if self._written > self._max_chars:
                    self.limit_exceeded = True
                    raise OSError("submission output limit exceeded")
                return super().write(value)

        _solution_code = {code!r}
        _test_cases = json.loads({cases_json!r})

        for _i, _case in enumerate(_test_cases):
            _input_data = str(_case.get("input", ""))
            _expected = str(_case.get("output", _case.get("expected_output", ""))).strip()

            sys.stdin = io.StringIO(_input_data)
            _captured = _LimitedStringIO({output_limit})
            sys.stdout = _captured

            try:
                exec(_solution_code, {{"__name__": "__main__"}})
            finally:
                sys.stdout = sys.__stdout__
                sys.stdin = sys.__stdin__

            if _captured.limit_exceeded:
                print(f"FAIL case {{_i}}: output limit exceeded", file=sys.stderr)
                sys.exit(1)

            _actual = _captured.getvalue().strip()
            if _actual != _expected:
                print(
                    f"FAIL case {{_i}}: expected {{_expected!r}}, got {{_actual!r}}",
                    file=sys.stderr,
                )
                sys.exit(1)
    """)

    cases_json = json.dumps(all_cases, ensure_ascii=False)
    formatted_script = runner_template.format(
        code=code,
        cases_json=cases_json,
        output_limit=1024 * 1024,
    )
    return f"{limits_header}\n\n{formatted_script}"



def _run_script(script: str, timeout: int = DEFAULT_TIMEOUT) -> bool:
    """Execute *script* in a subprocess and return True if it exits cleanly."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(script)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.debug("Execution timed out after %ss", timeout)
        return False
    except Exception as exc:
        logger.debug("Execution error: %s", exc)
        return False
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass




def _detect_dataset_type(reference: dict) -> str:
    """Determine whether reference came from HumanEval or LiveCodeBench."""
    if "entry_point" in reference and "test" in reference:
        return "humaneval"
    if "public_test_cases" in reference or "private_test_cases" in reference:
        return "livecodebench"
    # Default to HumanEval-style if it has a test field.
    if "test" in reference:
        return "humaneval"
    return "unknown"


def code_exec_equal(
    prediction: Union[str, None],
    reference: Union[str, dict],
    timeout: int = DEFAULT_TIMEOUT,
    test_type: str = "full",
) -> bool:
    """Evaluate whether the predicted code passes the reference test suite.

    Parameters
    ----------
    prediction:
        Raw model output text (may include ``<think>`` blocks and markdown
        fences).
    reference:
        Either a JSON string or a dict containing test harness data.  For
        HumanEval this includes ``test``, ``entry_point``, and
        ``canonical_solution``.  For LiveCodeBench this includes
        `public_test_cases``, ``private_test_cases``, and ``meta_data``.
    timeout:
        Per-problem execution timeout in seconds.
    test_type:
        Type of tests to evaluate: "public", "private", or "full".
    """
    if not prediction:
        return False

    code = extract_code_block(str(prediction))
    if not code:
        return False

    if isinstance(reference, str):
        try:
            reference = json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse reference as JSON: %s", reference[:200])
            return False

    if not isinstance(reference, dict):
        return False

    dataset_type = _detect_dataset_type(reference)

    if dataset_type == "humaneval":
        script = _build_humaneval_script(code, reference, test_type=test_type)
    elif dataset_type == "livecodebench":
        script = _build_livecodebench_script(code, reference, test_type=test_type)
    else:
        logger.warning("Unknown dataset type for code evaluation")
        return False

    return _run_script(script, timeout=timeout)


def code_exec_equal_public(
    prediction: Union[str, None],
    reference: Union[str, dict],
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """Evaluate prediction on public test cases only."""
    return code_exec_equal(prediction, reference, timeout=timeout, test_type="public")


def code_exec_equal_private(
    prediction: Union[str, None],
    reference: Union[str, dict],
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """Evaluate prediction on private test cases only."""
    return code_exec_equal(prediction, reference, timeout=timeout, test_type="private")


def code_exec_equal_full(
    prediction: Union[str, None],
    reference: Union[str, dict],
    timeout: int = DEFAULT_TIMEOUT,
) -> bool:
    """Evaluate prediction on full (public + private) test suite."""
    return code_exec_equal(prediction, reference, timeout=timeout, test_type="full")
