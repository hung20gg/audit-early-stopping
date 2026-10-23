from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Iterable

from methods.response import Response
from random_exit.sampling import metadata_path
from replay_eval.v2.config.schema import ReplayV2Config
from replay_eval.v2.engine.types import RequestKind, RequestSpec
from replay_eval.v2.rendering.hf import CanonicalRenderer


LOGGER = logging.getLogger(__name__)
ROW_FIELDS = {
    "key",
    "data_id",
    "line_number",
    "model",
    "position",
    "num_run",
    "output",
    "finish_reason",
    "answer_tokens",
    "ok",
}
OPTIONAL_CHECKPOINT_FIELDS = {"correct"}


def _stable_seed(*parts: Any) -> int:
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & 0x7FFFFFFF


def plan_positions(
    reasoning_tokens: int,
    *,
    position_seed: int,
    model: str,
    data_id: str,
    spacing: int,
    max_positions: int,
) -> list[int]:
    if reasoning_tokens < 1:
        return []
    rng = random.Random(_stable_seed(position_seed, model, data_id, "positions"))
    offset = rng.randint(1, spacing)
    candidates = list(range(offset, reasoning_tokens + 1, spacing))
    if len(candidates) > max_positions:
        candidates = rng.sample(candidates, max_positions)
    return sorted(candidates)


def rollout_seed(
    position_seed: int,
    model: str,
    data_id: str,
    position: int,
    num_run: int,
) -> int:
    return _stable_seed(
        position_seed, model, data_id, position, num_run, "continuation"
    )


def _row_key(data_id: str, position: int, num_run: int) -> str:
    return f"{data_id}_{position}_{num_run}"


def _checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_final = index == len(lines)
            terminated = raw.endswith((b"\n", b"\r"))
            if is_final and not terminated:
                LOGGER.warning("Ignoring truncated final JSONL line in %s", path)
                break
            raise ValueError(f"Invalid JSONL at line {index}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {index} is not an object")
        unknown = set(value) - ROW_FIELDS - OPTIONAL_CHECKPOINT_FIELDS
        missing = ROW_FIELDS - set(value)
        if unknown or missing:
            raise ValueError(
                f"Invalid random-exit row schema at line {index}: "
                f"missing={sorted(missing)}, extra={sorted(unknown)}"
            )
        rows.append(value)
    return rows


def _completed_checkpoint(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int]:
    completed: dict[str, dict[str, Any]] = {}
    failures = 0
    for row in rows:
        key = str(row["key"])
        if not row["ok"]:
            failures += 1
            continue
        previous = completed.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"Conflicting duplicate random-exit key: {key}")
        completed[key] = row
    return completed, failures


def _input_manifest_identity(input_file: Path, records: list[dict[str, Any]]) -> str:
    sidecar = metadata_path(input_file)
    if sidecar.is_file():
        value = json.loads(sidecar.read_text(encoding="utf-8"))
        identity = value.get("manifest_identity") if isinstance(value, dict) else None
        if identity:
            return str(identity)
    identities = sorted(
        (
            str(record.get("data_id")),
            int(record.get("line_number", 0)),
        )
        for record in records
    )
    payload = json.dumps(identities, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolved_settings(
    config: ReplayV2Config, *, manifest_identity: str
) -> dict[str, Any]:
    values = {
        "manifest_identity": manifest_identity,
        "model": {
            **asdict(config.model),
            "engine_kwargs": config.engine_kwargs(),
        },
        "random_exit": asdict(config.random_exit),
        "runtime": {
            "max_final_answer_completion_token": (
                config.runtime.max_final_answer_completion_token
            ),
            "sampling": config.runtime.sampling,
        },
        "rendering": {
            "system_prompt": config.rendering.system_prompt,
            "think_start_token": config.rendering.think_start_token,
            "think_end_token": config.rendering.think_end_token,
            "template_kwargs": config.rendering.template_kwargs,
        },
    }
    # Sidecar comparisons must not depend on tuple/list representation details.
    return json.loads(json.dumps(values, ensure_ascii=False, sort_keys=True))


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar = metadata_path(path)
    if not sidecar.is_file():
        raise ValueError(f"Random-exit checkpoint sidecar is missing: {sidecar}")
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Random-exit checkpoint sidecar is not an object: {sidecar}")
    return value


def validate_resume_metadata(
    config: ReplayV2Config,
    *,
    input_file: Path,
    records: list[dict[str, Any]],
    resume_from: Path | None,
) -> None:
    if resume_from is None:
        return
    if not resume_from.is_file():
        raise FileNotFoundError(f"Replay v2 checkpoint does not exist: {resume_from}")
    expected = resolved_settings(
        config,
        manifest_identity=_input_manifest_identity(input_file, records),
    )
    checkpoint = _load_sidecar(resume_from).get("resolved")
    if checkpoint != expected and not _is_manifest_extension(checkpoint, expected):
        raise ValueError(
            "Random-exit checkpoint metadata conflicts with resolved settings"
        )
    _checkpoint_rows(resume_from)


def _is_manifest_extension(checkpoint: Any, expected: dict[str, Any]) -> bool:
    """Allow an old checkpoint when only its permanent sample was extended."""
    if not isinstance(checkpoint, dict):
        return False
    previous = json.loads(json.dumps(checkpoint))
    current = json.loads(json.dumps(expected))
    previous_random = previous.get("random_exit")
    current_random = current.get("random_exit")
    if not isinstance(previous_random, dict) or not isinstance(current_random, dict):
        return False
    previous_size = previous_random.get("sample_size")
    current_size = current_random.get("sample_size")
    if (
        not isinstance(previous_size, int)
        or isinstance(previous_size, bool)
        or not isinstance(current_size, int)
        or isinstance(current_size, bool)
        or current_size <= previous_size
    ):
        return False
    previous.pop("manifest_identity", None)
    current.pop("manifest_identity", None)
    previous_random.pop("sample_size", None)
    current_random.pop("sample_size", None)
    return previous == current


def _write_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    sidecar = metadata_path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(sidecar)


def _repair_append_boundary(path: Path) -> None:
    data = path.read_bytes()
    if not data or data.endswith((b"\n", b"\r")):
        return
    final_start = data.rfind(b"\n") + 1
    try:
        json.loads(data[final_start:])
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(final_start)
        return
    with path.open("ab") as handle:
        handle.write(b"\n")


def _trajectory(record: dict[str, Any]) -> Response:
    value = (
        record.get("response")
        or record.get("trajectory")
        or record.get("generation_response")
    )
    if isinstance(value, dict):
        return Response(**value)
    if isinstance(record.get("output"), str):
        return Response(
            input=str(record.get("prompt_input") or ""),
            content=record["output"],
            finish_reason=record.get("finish_reason"),
            prompt_tokens=record.get("prompt_tokens"),
            completion_tokens=record.get("completion_tokens")
            or record.get("num_token"),
        )
    raise ValueError("Random-exit replay requires a supplied trajectory")


def _question(record: dict[str, Any]) -> str:
    nested = record.get("input")
    nested = nested if isinstance(nested, dict) else {}
    value = (
        record.get("question")
        or record.get("problem")
        or record.get("prompt")
        or nested.get("question")
        or nested.get("problem")
        or nested.get("prompt")
    )
    if not value:
        raise ValueError("Random-exit input has no question/problem/prompt")
    return str(value)


class RandomExitReplayRunner:
    def __init__(
        self,
        config: ReplayV2Config,
        *,
        engine: Any,
        renderer: CanonicalRenderer,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.renderer = renderer
        self.settings = settings

    def _messages(self, question: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.config.rendering.system_prompt:
            messages.append(
                {"role": "system", "content": self.config.rendering.system_prompt}
            )
        messages.append({"role": "user", "content": question})
        return messages

    def _prepare_record(
        self, record: dict[str, Any], index: int
    ) -> tuple[str, int, Response, list[int], list[int], list[int], str]:
        data_id = record.get("data_id")
        if not isinstance(data_id, str) or not data_id:
            raise ValueError(f"Random-exit input record {index} has no data_id")
        line_number = record.get("line_number")
        if (
            not isinstance(line_number, int)
            or isinstance(line_number, bool)
            or line_number < 1
        ):
            raise ValueError(
                f"Random-exit input record {index} has invalid line_number"
            )
        trajectory = _trajectory(record)
        messages = self._messages(_question(record))
        provenance = "stored"
        if not trajectory.input:
            trajectory.input = self.renderer.source_text(messages)
            provenance = "reconstructed"
        stored_output_ids = record.get("output_token_ids")
        output_ids = (
            [int(value) for value in stored_output_ids]
            if isinstance(stored_output_ids, list)
            else None
        )
        trace = self.renderer.full_trace(
            messages,
            trajectory.content,
            stored_input=trajectory.input,
            output_token_ids=output_ids,
        )
        output_ids = list(trace.output_token_ids)
        explicit_count = record.get(
            "reasoning_token_count", record.get("thinking_token_count")
        )
        reasoning_tokens = (
            int(explicit_count)
            if explicit_count is not None
            else int(trace.thinking_token_count)
        )
        reasoning_tokens = max(0, min(reasoning_tokens, len(output_ids)))
        model = self.config.model.key or self.config.model.id
        positions = plan_positions(
            reasoning_tokens,
            position_seed=self.config.random_exit.position_seed,
            model=model,
            data_id=data_id,
            spacing=self.config.random_exit.spacing,
            max_positions=self.config.random_exit.max_positions,
        )
        source_ids = list(trace.source_token_ids)
        return (
            data_id,
            line_number,
            trajectory,
            source_ids,
            output_ids,
            positions,
            provenance,
        )

    def _request(
        self,
        *,
        data_id: str,
        trajectory: Response,
        source_ids: list[int],
        output_ids: list[int],
        position: int,
        num_run: int,
    ) -> RequestSpec:
        content = self.renderer.tokenizer.decode(
            output_ids[:position],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix = Response(input=trajectory.input, content=content)
        # Keep Response.input authoritative. Calling this accessor is important:
        # follow-up generation always means the original rendered prompt plus
        # the fixed observed continuation prefix.
        generation_content = prefix.get_generation_content()
        if not generation_content.startswith(trajectory.input):
            raise ValueError("Random-exit generation prefix lost Response.input")
        suffix_ids = self.renderer.tokenizer.encode(
            f"You are out of thinking budget {self.config.rendering.think_end_token}",
            add_special_tokens=False,
        )
        prompt_ids = [*source_ids, *output_ids[:position], *suffix_ids]
        sampling = dict(self.config.runtime.sampling)
        sampling.pop("max_completion_token", None)
        sampling["max_tokens"] = self.config.runtime.max_final_answer_completion_token
        sampling["seed"] = rollout_seed(
            self.config.random_exit.position_seed,
            self.config.model.key or self.config.model.id,
            data_id,
            position,
            num_run,
        )
        return RequestSpec.create(
            prompt_ids,
            sampling,
            position,
            RequestKind.RANDOM_EXIT,
            example_id=data_id,
            cache_slot=num_run,
        )

    async def run(
        self,
        records: list[dict[str, Any]],
        input_file: Path,
        output_file: Path,
        *,
        resume_from: Path | None,
    ) -> dict[str, Any]:
        manifest_identity = _input_manifest_identity(input_file, records)
        settings = self.settings or resolved_settings(
            self.config, manifest_identity=manifest_identity
        )
        checkpoint_rows: list[dict[str, Any]] = []
        checkpoint_failures = 0
        if resume_from is not None:
            checkpoint_rows = _checkpoint_rows(resume_from)
        completed, checkpoint_failures = _completed_checkpoint(checkpoint_rows)

        LOGGER.info(
            "Random-exit preparing trajectories records=%d checkpoint_rows=%d "
            "completed_checkpoint_rows=%d checkpoint_failures=%d",
            len(records),
            len(checkpoint_rows),
            len(completed),
            checkpoint_failures,
        )

        prepared = [
            self._prepare_record(record, index)
            for index, record in enumerate(records, start=1)
        ]
        input_ids = [item[0] for item in prepared]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("Random-exit input contains duplicate data_id values")
        excluded = sum(
            len(item[5]) < self.config.random_exit.min_valid_positions
            for item in prepared
        )
        provenance = {
            "stored": sum(item[6] == "stored" for item in prepared),
            "reconstructed": sum(item[6] == "reconstructed" for item in prepared),
        }
        expected_rows: dict[str, tuple[str, int, str, int, int]] = {}
        output_model = self.config.model.key or self.config.model.id
        for data_id, line_number, _trajectory_value, _source, _output, positions, _ in prepared:
            if len(positions) < self.config.random_exit.min_valid_positions:
                continue
            for position in positions:
                for num_run in range(self.config.random_exit.num_runs):
                    key = _row_key(data_id, position, num_run)
                    expected_rows[key] = (
                        data_id,
                        line_number,
                        output_model,
                        position,
                        num_run,
                    )
        for key, row in completed.items():
            expected = expected_rows.get(key)
            actual = (
                row["data_id"],
                row["line_number"],
                row["model"],
                row["position"],
                row["num_run"],
            )
            if expected is None or actual != expected:
                raise ValueError(
                    f"Checkpoint row does not match the deterministic plan: {key}"
                )
        planned: list[
            tuple[str, int, Response, list[int], list[int], int, int]
        ] = []
        for data_id, line_number, trajectory, source_ids, output_ids, positions, _ in prepared:
            if len(positions) < self.config.random_exit.min_valid_positions:
                continue
            for position in positions:
                for num_run in range(self.config.random_exit.num_runs):
                    key = _row_key(data_id, position, num_run)
                    if key not in completed:
                        planned.append(
                            (
                                data_id,
                                line_number,
                                trajectory,
                                source_ids,
                                output_ids,
                                position,
                                num_run,
                            )
                        )
        sampled_positions = sum(
            len(item[5])
            for item in prepared
            if len(item[5]) >= self.config.random_exit.min_valid_positions
        )
        sampling = self.config.runtime.sampling
        LOGGER.info(
            "Random-exit plan model=%s trajectories=%d included=%d excluded=%d "
            "positions=%d runs_per_position=%d total_continuations=%d resumed=%d "
            "pending=%d concurrency=%d max_answer_tokens=%d temperature=%s top_p=%s",
            output_model,
            len(prepared),
            len(prepared) - excluded,
            excluded,
            sampled_positions,
            self.config.random_exit.num_runs,
            len(expected_rows),
            len(completed),
            len(planned),
            self.config.runtime.example_concurrency,
            self.config.runtime.max_final_answer_completion_token,
            sampling.get("temperature"),
            sampling.get("top_p"),
        )
        metadata = {
            "version": 1,
            "kind": "random_exit_replay",
            "resolved": settings,
            "prompt_provenance": provenance,
            "exclusion_counts": {
                "fewer_than_min_valid_positions": excluded,
                "included_trajectories": len(prepared) - excluded,
            },
            "failure_counts": {
                "checkpoint_rows": checkpoint_failures,
                "this_run": 0,
            },
            "continuations": {
                "completed_before_run": len(completed),
                "planned_this_run": len(planned),
                "completed_this_run": 0,
            },
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        append = resume_from == output_file
        if append:
            _repair_append_boundary(output_file)
        mode = "a" if append else "w"
        started = time.perf_counter()
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.config.runtime.example_concurrency)
        progress_total = len(planned)
        attempted = 0
        record_number = {
            data_id: index
            for index, (data_id, *_rest) in enumerate(prepared, start=1)
        }
        position_count = {item[0]: len(item[5]) for item in prepared}
        resumed_by_record = {data_id: 0 for data_id in input_ids}
        for row in completed.values():
            resumed_by_record[row["data_id"]] += 1
        pending_by_record = {data_id: 0 for data_id in input_ids}
        for plan in planned:
            pending_by_record[plan[0]] += 1
        attempted_by_record = {data_id: 0 for data_id in input_ids}
        succeeded_by_record = {data_id: 0 for data_id in input_ids}
        failed_by_record = {data_id: 0 for data_id in input_ids}
        last_error_by_record: dict[str, str] = {}

        LOGGER.info(
            "Random-exit writing output=%s metadata=%s mode=%s",
            output_file,
            metadata_path(output_file),
            "append" if append else "write",
        )

        for data_id in input_ids:
            if pending_by_record[data_id] != 0:
                continue
            is_excluded = (
                position_count[data_id]
                < self.config.random_exit.min_valid_positions
            )
            LOGGER.info(
                "Random-exit record input=%d/%d data_id=%s status=%s positions=%d "
                "resumed=%d generated=0 failed=0",
                record_number[data_id],
                len(prepared),
                data_id,
                "excluded" if is_excluded else "already_complete",
                position_count[data_id],
                resumed_by_record[data_id],
            )

        async def generate(
            plan: tuple[str, int, Response, list[int], list[int], int, int],
            handle: Any,
        ) -> None:
            nonlocal attempted
            (
                data_id,
                line_number,
                trajectory,
                source_ids,
                output_ids,
                position,
                num_run,
            ) = plan
            key = _row_key(data_id, position, num_run)
            row = {
                "key": key,
                "data_id": data_id,
                "line_number": line_number,
                "model": output_model,
                "position": position,
                "num_run": num_run,
                "output": "",
                "finish_reason": None,
                "answer_tokens": 0,
                "ok": False,
            }
            try:
                spec = self._request(
                    data_id=data_id,
                    trajectory=trajectory,
                    source_ids=source_ids,
                    output_ids=output_ids,
                    position=position,
                    num_run=num_run,
                )
                async with semaphore:
                    result = await self.engine.submit(spec)
                row.update(
                    {
                        "output": result.response.content,
                        "finish_reason": result.response.finish_reason,
                        "answer_tokens": result.response.completion_tokens
                        if result.response.completion_tokens is not None
                        else len(result.output_token_ids),
                        "ok": True,
                    }
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error_by_record[data_id] = (
                    f"{type(error).__name__}: {error}"
                )
            async with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if row["ok"]:
                    metadata["continuations"]["completed_this_run"] += 1
                else:
                    metadata["failure_counts"]["this_run"] += 1
                _write_sidecar(output_file, metadata)
                attempted += 1
                attempted_by_record[data_id] += 1
                if row["ok"]:
                    succeeded_by_record[data_id] += 1
                else:
                    failed_by_record[data_id] += 1
                if attempted_by_record[data_id] == pending_by_record[data_id]:
                    elapsed = time.perf_counter() - started
                    rate = attempted / elapsed if elapsed > 0 else 0.0
                    remaining = progress_total - attempted
                    eta_seconds = remaining / rate if rate > 0 else 0.0
                    message = (
                        "Random-exit record input=%d/%d data_id=%s status=%s "
                        "positions=%d resumed=%d generated=%d failed=%d "
                        "overall=%d/%d rate=%.2f continuations/s eta=%.1fs"
                    )
                    values = (
                        record_number[data_id],
                        len(prepared),
                        data_id,
                        "complete" if failed_by_record[data_id] == 0 else "failed",
                        position_count[data_id],
                        resumed_by_record[data_id],
                        succeeded_by_record[data_id],
                        failed_by_record[data_id],
                        attempted,
                        progress_total,
                        rate,
                        eta_seconds,
                    )
                    if failed_by_record[data_id]:
                        LOGGER.warning(
                            message + " last_error=%s",
                            *values,
                            last_error_by_record.get(data_id),
                        )
                    else:
                        LOGGER.info(message, *values)

        with output_file.open(mode, encoding="utf-8") as handle:
            if not append:
                for row in completed.values():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            _write_sidecar(output_file, metadata)
            # Keep a single trajectory hot in the prefix cache. Complete the
            # first replica before its siblings so their shared prefill is
            # available, then batch the remaining independent seeded answers.
            # Bounding tasks by one checkpoint also avoids creating hundreds
            # of thousands of coroutines for a full random-exit collection.
            from itertools import groupby

            for _checkpoint, batch in groupby(
                planned, key=lambda plan: (plan[0], plan[5])
            ):
                replicas = list(batch)
                await generate(replicas[0], handle)
                await asyncio.gather(
                    *(generate(plan, handle) for plan in replicas[1:])
                )
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "Random-exit finished generated=%d failed=%d resumed=%d elapsed=%.1fs "
            "output=%s metadata=%s",
            metadata["continuations"]["completed_this_run"],
            metadata["failure_counts"]["this_run"],
            len(completed),
            elapsed,
            output_file,
            metadata_path(output_file),
        )
        return {
            "random_exit": True,
            "trajectories": len(prepared),
            "excluded_trajectories": excluded,
            "resumed_continuations": len(completed),
            "generated_continuations": metadata["continuations"]["completed_this_run"],
            "failed_continuations": metadata["failure_counts"]["this_run"],
            "elapsed_seconds": elapsed,
        }
