from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from replay_eval.v2.engine.types import RequestResult, RequestKind


@dataclass
class RunMetrics:
    model_load_seconds: float = 0.0
    warmup_seconds: float = 0.0
    measured_seconds: float = 0.0
    rendering_alignment_seconds: float = 0.0
    hidden_extraction_seconds: float = 0.0
    hidden_loading_seconds: float = 0.0
    hidden_normalization_seconds: float = 0.0
    hidden_processing_seconds: float = 0.0
    terminator_seconds: float = 0.0
    thought_calibration_seconds: float = 0.0
    replay_seconds: float = 0.0
    final_prompt_build_seconds: float = 0.0
    evaluator_seconds: float = 0.0
    verifier_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    examples: int = 0
    failed_examples: int = 0
    timed_out_examples: int = 0
    resumed_examples: int = 0
    scan_tokens: int = 0
    probe_tokens: int = 0
    cached_tokens: int = 0
    computed_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    llm_call_seconds: float = 0.0
    peak_gpu_memory_bytes: int | None = None
    latencies: list[float] = field(default_factory=list)
    request_time: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_request(self, kind: RequestKind, result: RequestResult) -> None:
        bucket = self.request_time.setdefault(
            str(kind), {"queue": 0.0, "prefill": 0.0, "decode": 0.0, "end_to_end": 0.0}
        )
        for key in bucket:
            bucket[key] += float(getattr(result.timing, key))
        if not result.cached:
            self.llm_calls += 1
            self.llm_call_seconds += float(result.timing.end_to_end)
        self.computed_tokens += result.computed_tokens
        if result.cached:
            self.cached_tokens += len(result.prompt_token_ids) + len(
                result.output_token_ids
            )
        else:
            self.cached_tokens += result.cached_prompt_tokens
        if kind in {RequestKind.NORMAL_PROBE, RequestKind.DTSR}:
            self.probe_tokens += len(result.output_token_ids)

    def summary(self) -> dict[str, Any]:
        elapsed = self.measured_seconds
        sorted_latency = sorted(self.latencies)
        percentiles: dict[str, float] = {}
        if sorted_latency:
            percentiles["p50"] = _percentile(sorted_latency, 0.50)
            percentiles["p95"] = _percentile(sorted_latency, 0.95)
            percentiles["p99"] = _percentile(sorted_latency, 0.99)
        data = asdict(self)
        data.pop("latencies", None)
        data.update(
            {
                "examples_per_second": self.examples / elapsed if elapsed else 0.0,
                "scan_tokens_per_second": self.scan_tokens / elapsed
                if elapsed
                else 0.0,
                "probe_tokens_per_second": self.probe_tokens / elapsed
                if elapsed
                else 0.0,
                "cache_hit_ratio": self.cache_hits
                / (self.cache_hits + self.cache_misses)
                if self.cache_hits + self.cache_misses
                else 0.0,
                "avg_llm_call_seconds": self.llm_call_seconds / self.llm_calls
                if self.llm_calls
                else 0.0,
                "latency_seconds": percentiles,
            }
        )
        return data


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    index = fraction * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
