#!/usr/bin/env python3
"""Decode byte-level BPE text in replay outputs and method answers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2/DeepSeek byte-to-Unicode tokenizer mapping."""
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = visible[:]
    codepoints = visible[:]
    for byte in range(256):
        if byte not in visible:
            byte_values.append(byte)
            codepoints.append(256 + len(byte_values) - len(visible) - 1)
    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}


def decode_output(output: str, decoder: dict[str, int]) -> str:
    # Already decoded whitespace can coexist with token glyphs after a partial fix.
    has_token_glyph = any(
        ord(character) > 255 and character != "ħ" and character in decoder
        for character in output
    )
    has_latin1_mojibake = any(
        output[index] in "ÂÃ" and 0x80 <= decoder.get(output[index + 1], -1) <= 0xBF
        for index in range(len(output) - 1)
    )
    if not has_token_glyph and not has_latin1_mojibake:
        return output
    data = bytearray()
    for character in output:
        if character in decoder:
            data.append(decoder[character])
        else:
            data.extend(character.encode("utf-8"))
    # Some traces already contain literal Latin-1 characters (for example A°)
    # alongside token glyphs. Preserve bytes that cannot form UTF-8 sequences.
    decoded = data.decode("utf-8", errors="surrogateescape")
    return "".join(
        chr(ord(character) - 0xDC00) if 0xDC80 <= ord(character) <= 0xDCFF else character
        for character in decoded
    )


def decode_in_place(path: Path) -> tuple[int, int]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    rows = changed = 0
    decoder = byte_decoder()
    try:
        with path.open("r", encoding="utf-8") as source, os.fdopen(
            descriptor, "w", encoding="utf-8"
        ) as destination:
            for rows, line in enumerate(source, start=1):
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("output"), str):
                    raise ValueError(f"Expected a JSON object with string output at line {rows}")
                output = row["output"]
                row["output"] = decode_output(output, decoder)
                row_changed = row["output"] != output
                method_results = list(row.get("method_results") or [])
                method_results.extend(
                    (row.get("additional_logs") or {}).get("method_results") or []
                )
                for result in method_results:
                    answer = result.get("final_answer")
                    if isinstance(answer, str):
                        result["final_answer"] = decode_output(answer, decoder)
                        row_changed |= result["final_answer"] != answer
                changed += row_changed
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        shutil.copymode(path, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return rows, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL file to update in place")
    args = parser.parse_args()
    rows, changed = decode_in_place(args.path)
    print(f"Decoded output or method answers in {changed}/{rows} records: {args.path}")


if __name__ == "__main__":
    main()
