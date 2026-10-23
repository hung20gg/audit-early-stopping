#!/usr/bin/env python
"""Re-score top-level and method answers in a normal-run or replay JSONL file."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import import_config_entry, load_config, resolve_model_config
from scripts.runner import extract_output_for_scoring


def empty_last_box(text: str) -> bool:
    """Reject an empty final box even if an earlier box contains an answer."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return False
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                content = text[start:index]
                content = re.sub(r"\\[a-zA-Z]+", "", content)
                return not content.strip("{} $\\\t\r\n")
    return False


def score(text, answer, evaluator, think_end_token=None, evaluator_name=None) -> bool:
    if not isinstance(text, str):
        return False
    candidate = extract_output_for_scoring(text, think_end_token)
    if empty_last_box(candidate):
        return False
    if evaluator(candidate, answer):
        return True
    if evaluator_name in ("mmlu_pro", "mmlu_pro_subset"):
        # Some method answers return a bare option, optionally with a period.
        option = re.fullmatch(r"\s*([A-J])\.?\s*", candidate, re.I)
        boxed = re.fullmatch(r"\s*\\(?:boxed|fbox)\s*\{\s*([A-J])\.\s*\}\s*", candidate, re.I)
        if option or boxed:
            return bool(evaluator(rf"\boxed{{{(option or boxed).group(1)}}}", answer))
    return False


def rescore_row(
    row: dict, config: dict, evaluator_override: str | None = None,
    changes: list[dict] | None = None,
) -> Counter:
    counts = Counter()
    data = row.get("input")
    if isinstance(data, dict) and "answer" in data:
        answer = data["answer"]
    elif "answer" in row:
        answer = row["answer"]
    else:
        raise ValueError("Row has neither input.answer nor top-level answer")
    evaluator_name = evaluator_override or row.get("evaluator")
    if not evaluator_name:
        raise ValueError("Normal-run row has no evaluator; provide --evaluator")
    evaluator = import_config_entry(config, "evaluator", evaluator_name)
    counts[f"evaluator:{evaluator_name}"] += 1
    if evaluator_override:
        row["evaluator"] = evaluator_name
    model_name = row.get("model")
    model_config = resolve_model_config(config, model_name) if model_name else {}
    think_end_token = model_config.get("think_end_token")

    old = row.get("correct")
    row["correct"] = score(row.get("output"), answer, evaluator, think_end_token, evaluator_name)
    counts[f"top_{old}_to_{row['correct']}"] += 1
    if changes is not None and old != row["correct"]:
        changes.append({"line_number": row.get("line_number"), "method": "top-level", "before": old, "after": row["correct"], "evaluator": evaluator_name})

    logs = row.get("additional_logs")
    result_groups = [row.get("method_results")]
    if isinstance(logs, dict):
        result_groups.append(logs.get("method_results"))
    for results in result_groups:
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict) or "final_answer" not in result:
                counts["missing_method_answer"] += 1
                continue
            old = result.get("correct")
            result["correct"] = score(result["final_answer"], answer, evaluator,
                                      evaluator_name=evaluator_name)
            counts[f"method_{old}_to_{result['correct']}"] += 1
            if changes is not None and old != result["correct"]:
                changes.append({"line_number": row.get("line_number"), "method": result.get("method", "<unnamed>"), "before": old, "after": result["correct"], "evaluator": evaluator_name})
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "scripts/config.yml")
    parser.add_argument("--evaluator", help="Override each row's evaluator")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument("--in-place", action="store_true", help="Atomically replace the input file")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.in_place:
        parser.error("Specify --dry-run or --in-place")
    if not args.input.is_file():
        parser.error(f"Input file does not exist or is not a regular file: {args.input}")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    counts = Counter()
    changes = []
    temporary = None
    try:
        if args.in_place and not args.dry_run:
            descriptor, name = tempfile.mkstemp(dir=args.input.parent, prefix=f".{args.input.name}.", suffix=".tmp", text=True)
            temporary = Path(name)
            output = os.fdopen(descriptor, "w", encoding="utf-8")
        else:
            output = None
        with args.input.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    if output:
                        output.write(line)
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected JSON object")
                    counts.update(rescore_row(row, config, args.evaluator, changes))
                except (ValueError, KeyError) as error:
                    raise ValueError(f"{args.input}:{line_number}: {error}") from error
                counts["rows"] += 1
                if output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if output:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
            os.chmod(temporary, args.input.stat().st_mode)
            os.replace(temporary, args.input)
            temporary = None
    finally:
        if "output" in locals() and output and not output.closed:
            output.close()
        if temporary:
            temporary.unlink(missing_ok=True)
    for kind in ("top", "method"):
        before_true = sum(value for key, value in counts.items() if key.startswith(kind + "_True_to_"))
        after_true = sum(value for key, value in counts.items() if key.startswith(kind + "_") and key.endswith("_to_True"))
        before_false = sum(value for key, value in counts.items() if key.startswith(kind + "_False_to_"))
        after_false = sum(value for key, value in counts.items() if key.startswith(kind + "_") and key.endswith("_to_False"))
        print(f"{kind}: before true={before_true:,}, false={before_false:,}; after true={after_true:,}, false={after_false:,}; changed={counts[f'{kind}_False_to_True'] + counts[f'{kind}_True_to_False']:,}")
    print("evaluators:", ", ".join(f"{key.removeprefix('evaluator:')}={value:,}" for key, value in sorted(counts.items()) if key.startswith("evaluator:")))
    print(f"rows={counts['rows']:,}; missing method answers={counts['missing_method_answer']:,}")
    for change in changes:
        print(f"line {change['line_number']} {change['method']}: {change['before']} -> {change['after']} ({change['evaluator']})")


if __name__ == "__main__":
    main()
