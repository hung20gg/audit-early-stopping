"""Dataset-first preparation and deterministic random-exit replay helpers."""

from random_exit.sampling import (
    MANIFEST_VERSION,
    build_or_load_manifest,
    model_run_line_numbers,
    manifest_identity,
    prepare_model_run,
)

__all__ = [
    "MANIFEST_VERSION",
    "build_or_load_manifest",
    "model_run_line_numbers",
    "manifest_identity",
    "prepare_model_run",
]
