"""Resumable random-exit collection and causal, generation-free method replay.

The input is a sequence of records with ``data_id``, ``question``, a serialized
``response`` (including its original rendered ``input``), and
``output_token_ids``.  A callable ``grade(record, answer)`` supplies labels.
Generations are immutable; feature versions and replay results are derived data.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from methods.response import Response
from replay_eval.method import EarlyExitMethod, VerifierContext
from replay_eval.replay import _resolve_methods
from replay_eval.v2.engine.types import RequestKind, RequestSpec
from replay_eval.v2.runtime.adapter import ReplayCompatibilityAdapter


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _response_data(response: Response) -> dict[str, Any]:
    return response.model_dump() if hasattr(response, "model_dump") else response.dict()


@dataclass(frozen=True)
class SamplingSettings:
    sample_size: int | None = None
    sample_seed: int = 0
    position_seed: int = 0
    spacing: int = 16
    max_positions: int | None = None
    exit_repetitions: int = 2
    trial_repetitions: int = 1
    dense_audit_traces: int = 0
    answer_budget: int = 3072
    trial_budget: int = 256
    retain_trial_states: bool = False
    retain_reasoning_states: bool = False
    hidden_width: int = 2880
    pooled_vectors_per_probe: int = 0
    pooled_width: int = 2880
    output_head_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("spacing", "exit_repetitions", "trial_repetitions", "answer_budget", "trial_budget", "hidden_width", "pooled_width"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("sample_size", "max_positions"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")
        if self.dense_audit_traces < 0 or self.pooled_vectors_per_probe < 0:
            raise ValueError("audit and pooled-vector counts must be nonnegative")


@dataclass(frozen=True)
class PilotMeasurements:
    mean_answer_tokens: float
    generated_tokens_per_second: float
    prefill_tokens_per_second: float
    bytes_per_artifact: float
    prefill_cache_reuse: float = 0.0
    peak_gpu_memory_bytes: int | None = None
    peak_cpu_memory_bytes: int | None = None


@dataclass(frozen=True)
class AdvisoryThresholds:
    storage_bytes: int | None = None
    gpu_hours: float | None = None
    peak_gpu_memory_bytes: int | None = None
    peak_cpu_memory_bytes: int | None = None
    generation_calls: int | None = None


class ArtifactTrialAdapter(ReplayCompatibilityAdapter):
    """The standard verifier facade, populated only by exact saved trials."""

    def __init__(self, *args: Any, artifacts: dict[tuple[str, int, int], dict[str, Any]], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.artifacts = artifacts

    def load_trial(self, identity: str, position: int, slot: int, prefix: Response) -> None:
        row = self.artifacts.get((identity, position, slot))
        if row is None:
            raise KeyError(f"Missing exact trial artifact at position={position}, slot={slot}")
        self.put_temp_answer(position, Response(**row["response"]), slot, prefix=prefix)


class SampledExitEstimator:
    """Estimate a stop policy from independent answer draws at its exact stop."""

    def __init__(self, *, nearby_radius: int = 0):
        self.nearby_radius = nearby_radius

    @staticmethod
    def _summary(rows: list[dict[str, Any]], position: int, *, selected_slot: int | None = None) -> dict[str, Any]:
        if selected_slot is not None:
            rows = [row for row in rows if row["slot"] == selected_slot]
        if not rows:
            return {"supported": False, "reason": "unsampled_position_or_slot", "position": position}
        labels = [float(row["correct"]) for row in rows]
        n = len(labels)
        mean = sum(labels) / n
        # Wilson 95% interval remains meaningful for n=1 and boundary rates.
        z = 1.96
        denominator = 1 + z * z / n
        center = (mean + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / denominator
        return {
            "supported": True, "position": position, "samples": n,
            "estimated_correctness": mean,
            "confidence_interval_95": [max(0.0, center - radius), min(1.0, center + radius)],
            "mean_answer_tokens": sum(row["answer_tokens"] for row in rows) / n,
            "mean_generated_tokens": sum(row["answer_tokens"] for row in rows) / n,
            "mean_prefill_tokens": sum(row["prefill_tokens"] for row in rows) / n,
            "selected_slot": selected_slot,
        }

    def estimate(self, position: int, exits: Iterable[dict[str, Any]], *, selected_slot: int | None = None, audit_positions: set[int] | None = None) -> dict[str, Any]:
        rows = list(exits)
        exact = self._summary([row for row in rows if row["position"] == position], position, selected_slot=selected_slot)
        result = {"exact": exact}
        if self.nearby_radius and not exact["supported"]:
            nearby = [row for row in rows if abs(row["position"] - position) <= self.nearby_radius]
            result["nearby"] = self._summary(nearby, position, selected_slot=selected_slot)
            result["nearby"]["audit_checked"] = bool(audit_positions and position in audit_positions)
        return result

    def audit(self, decisions: Iterable[dict[str, Any]], exits: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Compare leave-position-out nearby estimates with exact dense samples."""
        rows = list(exits)
        errors = []
        for decision in decisions:
            position = int(decision["position"])
            exact = self._summary([row for row in rows if row["position"] == position], position)
            nearby = self._summary([row for row in rows if row["position"] != position and abs(row["position"] - position) <= self.nearby_radius], position)
            if exact["supported"] and nearby["supported"]:
                errors.append(abs(exact["estimated_correctness"] - nearby["estimated_correctness"]))
        return {"compared_positions": len(errors), "mean_absolute_error": sum(errors) / len(errors) if errors else None}


class RandomExitReplayEvaluation:
    def __init__(
        self, records: Iterable[dict[str, Any]], *, renderer: Any, artifact_dir: str | Path,
        sampling: SamplingSettings = SamplingSettings(), decoder: dict[str, Any] | None = None,
        engine: Any = None, grade: Callable[[dict[str, Any], str], bool] | None = None,
        pilot: PilotMeasurements | None = None, thresholds: AdvisoryThresholds = AdvisoryThresholds(),
    ) -> None:
        self.records = list(records)
        ids = [str(row["data_id"]) for row in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate data_id")
        self.renderer, self.engine, self.grade = renderer, engine, grade
        self.root = Path(artifact_dir)
        self.sampling, self.decoder, self.pilot, self.thresholds = sampling, dict(decoder or {}), pilot, thresholds
        self._planned: dict[str, Any] | None = None
        self._collection_summary: dict[str, Any] | None = None

    def _trace(self, record: dict[str, Any]) -> tuple[Response, list[int], int]:
        value = record.get("response") or record.get("trajectory")
        response = value if isinstance(value, Response) else Response(**value)
        if not response.input:
            raise ValueError(f"Trace {record['data_id']} lacks the original rendered Response.input")
        ids = [int(token) for token in record["output_token_ids"]]
        reasoning = min(int(record.get("reasoning_token_count", len(ids))), len(ids))
        return response, ids, reasoning

    def _selected(self) -> list[dict[str, Any]]:
        rows = sorted(self.records, key=lambda row: str(row["data_id"]))
        size = self.sampling.sample_size
        if size is not None and size < len(rows):
            rows = sorted(random.Random(self.sampling.sample_seed).sample(rows, size), key=lambda row: str(row["data_id"]))
        return rows

    def _positions(self, record: dict[str, Any], reasoning: int, audit_ids: set[str]) -> list[int]:
        if str(record["data_id"]) in audit_ids:
            return list(range(1, reasoning + 1))
        rng = random.Random(int(_hash([self.sampling.position_seed, record["data_id"]])[:16], 16))
        offset = rng.randint(1, self.sampling.spacing)
        positions = list(range(offset, reasoning + 1, self.sampling.spacing))
        limit = self.sampling.max_positions
        if limit is not None and len(positions) > limit:
            positions = sorted(rng.sample(positions, limit))
        return positions

    def _schedule(self) -> tuple[list[dict[str, Any]], set[str]]:
        selected = self._selected()
        count = min(self.sampling.dense_audit_traces, len(selected))
        audit_ids = set(random.Random(self.sampling.sample_seed + 1).sample([str(r["data_id"]) for r in selected], count))
        return selected, audit_ids

    def _spec(self, record: dict[str, Any], position: int, slot: int, kind: RequestKind) -> tuple[str, RequestSpec]:
        response, ids, _ = self._trace(record)
        prefix_text = self.renderer.tokenizer.decode(ids[:position], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prefix = Response(input=response.input, content=prefix_text)
        prompt = prefix.get_generation_content()
        if kind == RequestKind.NORMAL_PROBE:
            adapter = ReplayCompatibilityAdapter(self.renderer.tokenizer, self.renderer)
            prompt += adapter.temp_answer_suffix(record.get("kwargs") or {})
        else:
            prompt += f"You are out of thinking budget {self.renderer.think_end_token}"
        prompt_ids = self.renderer.tokenizer.encode(prompt, add_special_tokens=False)
        sampling = dict(self.decoder)
        sampling["max_tokens"] = self.sampling.trial_budget if kind == RequestKind.NORMAL_PROBE else self.sampling.answer_budget
        sampling["seed"] = int(_hash([record["data_id"], position, kind.value, slot, self.sampling.position_seed])[:8], 16)
        identity = _hash([record["data_id"], position, prompt_ids, sampling, slot, kind.value])
        return identity, RequestSpec.create(prompt_ids, sampling, position, kind, example_id=str(record["data_id"]), cache_slot=slot)

    def _artifact_path(self, identity: str) -> Path:
        return self.root / "generations" / identity[:2] / f"{identity}.json"

    def plan(self) -> dict[str, Any]:
        selected, audit_ids = self._schedule()
        positions = 0
        prefill = 0
        reasoning_tokens = 0
        for record in selected:
            _, _, reasoning = self._trace(record)
            reasoning_tokens += reasoning
            for position in self._positions(record, reasoning, audit_ids):
                positions += 1
                for kind, repeats in ((RequestKind.RANDOM_EXIT, self.sampling.exit_repetitions), (RequestKind.NORMAL_PROBE, self.sampling.trial_repetitions)):
                    for slot in range(repeats):
                        _, spec = self._spec(record, position, slot, kind)
                        prefill += len(spec.prompt_token_ids)
        exit_calls = positions * self.sampling.exit_repetitions
        trial_calls = positions * self.sampling.trial_repetitions
        calls = exit_calls + trial_calls
        maximum_tokens = exit_calls * self.sampling.answer_budget + trial_calls * self.sampling.trial_budget
        state_bytes = 2 * self.sampling.hidden_width
        storage = {
            "reasoning_states": reasoning_tokens * state_bytes if self.sampling.retain_reasoning_states else 0,
            "trial_states": trial_calls * self.sampling.trial_budget * state_bytes if self.sampling.retain_trial_states else 0,
            "pooled_features": calls * self.sampling.pooled_vectors_per_probe * self.sampling.pooled_width * 2,
            "output_head_shared": self.sampling.output_head_bytes,
            "generated_text": int(calls * (self.pilot.bytes_per_artifact if self.pilot else 1024) * 0.5),
            "metadata_and_token_ids": int(calls * (self.pilot.bytes_per_artifact if self.pilot else 1024) * 0.5),
        }
        projected_bytes = sum(storage.values())
        gpu_hours = None
        if self.pilot:
            generated = exit_calls * min(self.sampling.answer_budget, self.pilot.mean_answer_tokens) + trial_calls * min(self.sampling.trial_budget, self.pilot.mean_answer_tokens)
            gpu_hours = (generated / self.pilot.generated_tokens_per_second + prefill * (1 - self.pilot.prefill_cache_reuse) / self.pilot.prefill_tokens_per_second) / 3600
        warnings = []
        for name, value, limit in (("storage_bytes", projected_bytes, self.thresholds.storage_bytes), ("gpu_hours", gpu_hours, self.thresholds.gpu_hours), ("generation_calls", calls, self.thresholds.generation_calls), ("peak_gpu_memory_bytes", self.pilot.peak_gpu_memory_bytes if self.pilot else None, self.thresholds.peak_gpu_memory_bytes), ("peak_cpu_memory_bytes", self.pilot.peak_cpu_memory_bytes if self.pilot else None, self.thresholds.peak_cpu_memory_bytes)):
            if value is not None and limit is not None and value > limit:
                warnings.append(f"Projected {name} {value:g} exceeds advisory threshold {limit:g}")
        result = {"traces": len(selected), "audit_traces": len(audit_ids), "sampled_positions": positions, "exit_calls": exit_calls, "trial_calls": trial_calls, "requested_prefill_tokens": prefill, "maximum_generated_tokens": maximum_tokens, "projected_storage_bytes": projected_bytes, "storage_breakdown_bytes": storage, "estimated_gpu_hours": gpu_hours, "pilot": asdict(self.pilot) if self.pilot else None, "warnings": warnings, "sampling": asdict(self.sampling)}
        self._planned = result
        _dump(self.root / "manifest.json", result)
        return result

    async def collect_missing(self) -> dict[str, Any]:
        if self.engine is None:
            raise ValueError("Collection requires an engine")
        if self._planned is None:
            self.plan()
        selected, audit_ids = self._schedule()
        collected = skipped = 0
        for record in selected:
            response, ids, reasoning = self._trace(record)
            trace_path = self.root / "traces" / f"{_hash(record['data_id'])}.json"
            if not trace_path.exists():
                trace_row = {"data_id": record["data_id"], "question": record["question"], "response": _response_data(response), "output_token_ids": ids, "reasoning_token_count": reasoning}
                if self.sampling.retain_reasoning_states:
                    states = record.get("reasoning_states")
                    if states is None or len(states) != reasoning:
                        raise ValueError("retained reasoning states were requested but are absent from the input trace")
                    trace_row["reasoning_states"] = states
                _dump(trace_path, trace_row)
            for position in self._positions(record, reasoning, audit_ids):
                for kind, repeats in ((RequestKind.RANDOM_EXIT, self.sampling.exit_repetitions), (RequestKind.NORMAL_PROBE, self.sampling.trial_repetitions)):
                    for slot in range(repeats):
                        identity, spec = self._spec(record, position, slot, kind)
                        path = self._artifact_path(identity)
                        if path.exists():
                            skipped += 1
                            continue
                        result = await self.engine.submit(spec)
                        answer = result.response.content
                        correct = self.grade(record, answer) if self.grade and kind == RequestKind.RANDOM_EXIT else None
                        row = {"identity": identity, "data_id": record["data_id"], "position": position, "slot": slot, "kind": kind.value, "response": _response_data(result.response), "output_token_ids": result.output_token_ids, "answer_tokens": len(result.output_token_ids), "prefill_tokens": len(spec.prompt_token_ids), "cached_prefill_tokens": result.cached_prompt_tokens, "correct": correct, "timing": asdict(result.timing)}
                        if kind == RequestKind.NORMAL_PROBE and self.sampling.retain_trial_states:
                            states = getattr(result.raw, "hidden_states", None)
                            if states is None:
                                raise ValueError("retained trial states were requested but the engine returned none")
                            row["trial_states"] = states.tolist() if hasattr(states, "tolist") else states
                        _dump(path, row)
                        collected += 1
        self._collection_summary = {"collected": collected, "reused": skipped, "artifact_bytes": sum(p.stat().st_size for p in self.root.rglob("*.json")), "peak_gpu_memory_bytes": self.engine.peak_gpu_memory_bytes() if callable(getattr(self.engine, "peak_gpu_memory_bytes", None)) else None}
        return self._collection_summary

    def _artifacts(self, record: dict[str, Any], kind: RequestKind) -> list[dict[str, Any]]:
        _, _, reasoning = self._trace(record)
        _, audit_ids = self._schedule()
        repeats = self.sampling.exit_repetitions if kind == RequestKind.RANDOM_EXIT else self.sampling.trial_repetitions
        rows = []
        for position in self._positions(record, reasoning, audit_ids):
            for slot in range(repeats):
                identity, _ = self._spec(record, position, slot, kind)
                path = self._artifact_path(identity)
                if path.exists():
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row.get("identity") != identity or row.get("kind") != kind.value or row.get("slot") != slot or row.get("position") != position:
                        raise ValueError(f"Artifact identity mismatch: {path}")
                    rows.append(row)
        return rows

    def prepare_features(self, version: str, extractor: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        """Derive versioned compact features without regenerating answers."""
        path = self.root / "features" / f"{version}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        features = {}
        for record in self._selected():
            for kind in (RequestKind.RANDOM_EXIT, RequestKind.NORMAL_PROBE):
                for row in self._artifacts(record, kind):
                    try:
                        features[row["identity"]] = extractor(row)
                    except KeyError as exc:
                        raise ValueError("Feature needs discarded states; extract hidden states from saved token IDs") from exc
        _dump(path, features)
        return features

    def replay(self, methods: Iterable[EarlyExitMethod] | Iterable[dict[str, Any]], *, kwargs: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run checkpoint, verifier and stop rule causally, with no engine calls."""
        resolved = _resolve_methods(methods)
        results = []
        for record in self._selected():
            response, ids, reasoning = self._trace(record)
            trials = self._artifacts(record, RequestKind.NORMAL_PROBE)
            by_position = {(row["position"], row["slot"]): row for row in trials}
            adapter = ArtifactTrialAdapter(self.renderer.tokenizer, self.renderer, artifacts={(_hash([record["data_id"], row["position"]]), row["position"], row["slot"]): row for row in trials})
            active = list(resolved)
            for method in active:
                method.reset()
            prefix_ids = []
            for position in range(1, reasoning + 1):
                prefix_ids.append(ids[position - 1])
                content = self.renderer.tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                prefix = Response(input=response.input, content=content, logprobs=response.logprobs[:position], completion_tokens=position)
                for method in list(active):
                    ctx = VerifierContext(question=str(record["question"]), prefix=prefix, latest_token=self.renderer.tokenizer.decode([ids[position - 1]]), position=position, agent=adapter, kwargs={**(record.get("kwargs") or {}), **(kwargs or {})}, state=method.state, logs=method.logs, trajectory=None, method_name=method.name)
                    if not method.should_check(ctx):
                        continue
                    if (position, 0) in by_position:
                        adapter.load_trial(_hash([record["data_id"], position]), position, 0, prefix)
                    try:
                        score = method.verify(ctx)
                    except (KeyError, RuntimeError) as exc:
                        results.append({"data_id": record["data_id"], "method": method.name, "position": position, "supported": False, "reason": str(exc)})
                        active.remove(method)
                        continue
                    if method.should_stop(score, ctx):
                        results.append({"data_id": record["data_id"], "method": method.name, "position": 0 if method.no_thinking else position, "stopped_early": True, "verifier_score": score if isinstance(score, (int, float)) else None, "verifier_prompt_tokens": int(method.logs.get("verifier_prompt_tokens", 0)), "verifier_completion_tokens": int(method.logs.get("verifier_completion_tokens", 0))})
                        active.remove(method)
                if not active:
                    break
            for method in active:
                ending = response.content.rfind(self.renderer.think_end_token)
                final_answer = response.content[ending + len(self.renderer.think_end_token):] if ending >= 0 else ""
                results.append({"data_id": record["data_id"], "method": method.name, "position": reasoning, "stopped_early": False, "final_answer": final_answer, "verifier_prompt_tokens": int(method.logs.get("verifier_prompt_tokens", 0)), "verifier_completion_tokens": int(method.logs.get("verifier_completion_tokens", 0))})
        return results

    def estimate(self, replay_rows: Iterable[dict[str, Any]], *, nearby_radius: int = 0) -> list[dict[str, Any]]:
        estimator = SampledExitEstimator(nearby_radius=nearby_radius)
        lookup = {str(record["data_id"]): record for record in self._selected()}
        _, audit_ids = self._schedule()
        estimates = []
        for decision in replay_rows:
            record = lookup[str(decision["data_id"])]
            if decision.get("changed_reasoning_continuation"):
                estimates.append({**decision, "exact": {"supported": False, "reason": "changed_reasoning_requires_paired_live_run"}})
                continue
            if not decision.get("supported", True):
                estimates.append({**decision, "exact": {"supported": False, "reason": decision.get("reason")}})
                continue
            if not decision.get("stopped_early", False):
                if self.grade is None:
                    result = {"supported": False, "reason": "missing_grader"}
                else:
                    result = {"supported": True, "estimated_correctness": float(self.grade(record, decision["final_answer"])), "samples": 1, "mean_answer_tokens": max(0, len(record["output_token_ids"]) - decision["position"])}
                estimates.append({**decision, "exact": result})
                continue
            exits = self._artifacts(record, RequestKind.RANDOM_EXIT)
            if self.grade:
                for row in exits:
                    if row["correct"] is None:
                        row["correct"] = self.grade(record, row["response"]["content"])
            if any(row["correct"] is None for row in exits):
                estimates.append({**decision, "exact": {"supported": False, "reason": "missing_labels"}})
                continue
            estimated = estimator.estimate(decision["position"], exits, selected_slot=decision.get("selected_slot"), audit_positions=set(range(1, self._trace(record)[2] + 1)) if record["data_id"] in audit_ids else set())
            estimates.append({**decision, **estimated})
        return estimates

    @staticmethod
    def validate_paired_live(estimated_rows: Iterable[dict[str, Any]], live_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Compare held-out paired live outcomes keyed by (trace, method)."""
        estimated = {(str(row["data_id"]), row["method"]): row for row in estimated_rows if row.get("exact", {}).get("supported")}
        live = {(str(row["data_id"]), row["method"]): row for row in live_rows}
        common = estimated.keys() & live.keys()
        errors = [abs(estimated[key]["exact"]["estimated_correctness"] - float(live[key]["correct"])) for key in common]
        names = sorted({key[1] for key in common})
        ranking = {}
        for name in names:
            keys = [key for key in common if key[1] == name]
            ranking[name] = {"estimated_accuracy": sum(estimated[key]["exact"]["estimated_correctness"] for key in keys) / len(keys), "live_accuracy": sum(float(live[key]["correct"]) for key in keys) / len(keys), "paired_cases": len(keys)}
        return {"paired_cases": len(common), "mean_absolute_error": sum(errors) / len(errors) if errors else None, "method_ranking": ranking, "estimated_order": sorted(names, key=lambda name: ranking[name]["estimated_accuracy"], reverse=True), "live_order": sorted(names, key=lambda name: ranking[name]["live_accuracy"], reverse=True)}

    def report(self, estimated_rows: Iterable[dict[str, Any]], *, paired_live_rows: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = list(estimated_rows)
        planned = self._planned or self.plan()
        methods = {}
        for method in sorted({row["method"] for row in rows}):
            group = [row for row in rows if row["method"] == method]
            supported = [row for row in group if row.get("exact", {}).get("supported")]
            methods[method] = {"cases": len(group), "supported": len(supported), "unsupported": len(group) - len(supported), "sampled_position_coverage": len(supported) / len(group) if group else 0, "estimated_accuracy": sum(row["exact"]["estimated_correctness"] for row in supported) / len(supported) if supported else None, "mean_compute_tokens": sum(row["position"] + row["exact"].get("mean_answer_tokens", 0) + row.get("verifier_prompt_tokens", 0) + row.get("verifier_completion_tokens", 0) for row in supported) / len(supported) if supported else None}
        artifacts = list((self.root / "generations").rglob("*.json")) if (self.root / "generations").exists() else []
        artifact_rows = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts]
        return {"methods": methods, "sampled_positions": planned["sampled_positions"], "generated_tokens": sum(row["answer_tokens"] for row in artifact_rows), "prefill_tokens": sum(row["prefill_tokens"] for row in artifact_rows), "cached_prefill_tokens": sum(row["cached_prefill_tokens"] for row in artifact_rows), "artifact_bytes": sum(path.stat().st_size for path in self.root.rglob("*.json")), "peak_gpu_memory_bytes": (self._collection_summary or {}).get("peak_gpu_memory_bytes") or (self.pilot.peak_gpu_memory_bytes if self.pilot else None), "peak_cpu_memory_bytes": self.pilot.peak_cpu_memory_bytes if self.pilot else None, "warnings": planned["warnings"], "paired_live_validation": self.validate_paired_live(rows, paired_live_rows) if paired_live_rows is not None else None}

    def pilot_measurements(self, *, peak_gpu_memory_bytes: int | None = None, peak_cpu_memory_bytes: int | None = None) -> PilotMeasurements:
        """Summarize a collected server pilot for projections of later runs."""
        paths = list((self.root / "generations").rglob("*.json"))
        if not paths:
            raise ValueError("No collected artifacts are available for a pilot")
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        generated = sum(row["answer_tokens"] for row in rows)
        prefill = sum(row["prefill_tokens"] for row in rows)
        elapsed = sum(row["timing"]["end_to_end"] for row in rows)
        prefill_seconds = sum(row["timing"]["prefill"] for row in rows)
        return PilotMeasurements(
            mean_answer_tokens=generated / len(rows),
            generated_tokens_per_second=generated / max(elapsed - prefill_seconds, 1e-9),
            prefill_tokens_per_second=prefill / max(prefill_seconds, 1e-9),
            bytes_per_artifact=sum(path.stat().st_size for path in paths) / len(paths),
            prefill_cache_reuse=sum(row["cached_prefill_tokens"] for row in rows) / prefill if prefill else 0,
            peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            peak_cpu_memory_bytes=peak_cpu_memory_bytes,
        )
