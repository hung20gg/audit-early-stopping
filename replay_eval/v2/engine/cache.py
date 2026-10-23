from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from replay_eval.v2.engine.types import RequestResult, RequestSpec, copy_response


class RequestCache:
    """Deduplicate exact in-flight and completed requests.

    The checkpoint and probe kind are part of ``RequestSpec``.  Consequently a
    normal temporary answer can only be reused at the same replay position and
    can never alias the full trace or the DTSR replacement prompt.
    """

    def __init__(self) -> None:
        self._values: dict[RequestSpec, RequestResult] = {}
        self._inflight: dict[RequestSpec, asyncio.Task[RequestResult]] = {}
        self.hits = 0
        self.misses = 0

    async def get_or_submit(
        self,
        spec: RequestSpec,
        submit: Callable[[RequestSpec], Awaitable[RequestResult]],
    ) -> RequestResult:
        cached = self._values.get(spec)
        if cached is not None:
            self.hits += 1
            return self._copy_hit(cached)
        task = self._inflight.get(spec)
        joined_inflight = task is not None
        if task is None:
            self.misses += 1
            task = asyncio.create_task(submit(spec))
            self._inflight[spec] = task
        else:
            self.hits += 1
        try:
            result = await task
        except BaseException:
            if self._inflight.get(spec) is task:
                self._inflight.pop(spec, None)
            raise
        self._values[spec] = result
        self._inflight.pop(spec, None)
        return self._copy_hit(result) if joined_inflight else result

    def get(self, spec: RequestSpec) -> RequestResult | None:
        value = self._values.get(spec)
        return self._copy_hit(value) if value is not None else None

    @staticmethod
    def _copy_hit(result: RequestResult) -> RequestResult:
        return RequestResult(
            response=copy_response(result.response),
            prompt_token_ids=list(result.prompt_token_ids),
            output_token_ids=list(result.output_token_ids),
            timing=result.timing,
            cached=True,
            cached_prompt_tokens=len(result.prompt_token_ids),
            raw=result.raw,
        )

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self) -> None:
        self._values.clear()
        self._inflight.clear()
        self.hits = self.misses = 0
