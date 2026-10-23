"""Random-exit continuations through the synchronous v1 completion client."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from llm import get_llm_wrapper
from methods.response import Response
from replay_eval.v2.engine.types import RequestResult, RequestSpec
from replay_eval.v2.rendering.hf import CanonicalRenderer
from replay_eval.v2.runtime.random_exit import (
    RandomExitReplayRunner,
    _input_manifest_identity,
    resolved_settings,
)


LOGGER = logging.getLogger(__name__)


class CompletionEngine:
    """Adapt v1 text completions to the shared random-exit planner and writer."""

    def __init__(self, llm: Any, tokenizer: Any) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_texts: dict[RequestSpec, str] = {}

    async def submit(self, spec: RequestSpec) -> RequestResult:
        return await asyncio.to_thread(self._complete, spec)

    def _complete(self, spec: RequestSpec) -> RequestResult:
        prompt = self.prompt_texts.pop(spec)
        sampling = spec.sampling_dict()
        max_tokens = sampling.pop("max_tokens")
        skip_special_tokens = sampling.pop("skip_special_tokens", None)
        if skip_special_tokens is not None:
            sampling["extra_body"] = {
                "skip_special_tokens": skip_special_tokens,
            }
        # The completion client takes a rendered text prompt, as in v1 replay.
        value = self.llm.completion(
            prompt,
            max_completion_token=max_tokens,
            detail=True,
            **sampling,
        )
        if value is None:
            raise RuntimeError("V1 completion returned no response")
        response = Response.from_completion(value)
        output_ids = self.tokenizer.encode(
            response.content, add_special_tokens=False
        )
        if response.completion_tokens is None:
            response.completion_tokens = len(output_ids)
        return RequestResult(
            response=response,
            prompt_token_ids=list(spec.prompt_token_ids),
            output_token_ids=list(output_ids),
        )


class V1RandomExitReplayRunner(RandomExitReplayRunner):
    def _request(self, **kwargs: Any) -> RequestSpec:
        spec = super()._request(**kwargs)
        trajectory = kwargs["trajectory"]
        output_ids = kwargs["output_ids"]
        position = kwargs["position"]
        prefix = Response(
            input=trajectory.input,
            content=self.renderer.tokenizer.decode(
                output_ids[:position],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )
        suffix = f"You are out of thinking budget {self.config.rendering.think_end_token}"
        if not isinstance(self.engine, CompletionEngine):
            raise TypeError("V1 random-exit requires CompletionEngine")
        self.engine.prompt_texts[spec] = prefix.get_generation_content() + suffix
        return spec


async def run_random_exit(
    config: Any,
    *,
    input_file: str | Path,
    output_file: str | Path,
    resume_from: str | Path | None,
    model_config: dict[str, Any],
    llm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    input_file = Path(input_file)
    output_file = Path(output_file)
    resume_path = Path(resume_from) if resume_from is not None else None
    with input_file.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    settings = resolved_settings(
        config, manifest_identity=_input_manifest_identity(input_file, records)
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Random-exit checkpoint does not exist: {resume_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.id)
    renderer = CanonicalRenderer(
        tokenizer,
        model_id=config.model.id,
        template_kwargs=config.rendering.template_kwargs,
        think_start_token=config.rendering.think_start_token,
        think_end_token=config.rendering.think_end_token,
    )
    llm = get_llm_wrapper(model_config["id"], **llm_kwargs)
    LOGGER.info("Random-exit v1 completion backend model=%s", model_config["id"])
    return await V1RandomExitReplayRunner(
        config,
        engine=CompletionEngine(llm, tokenizer),
        renderer=renderer,
        settings=settings,
    ).run(records, input_file, output_file, resume_from=resume_path)
