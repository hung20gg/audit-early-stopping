from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.response import LogProb, Response, TopLogProb
from replay_eval.checkpoints import AlwaysCheckpoint, EveryNTokens
from replay_eval.method import EarlyExitMethod, VerifierContext
from replay_eval.stop_rules import ThresholdGE
from replay_eval.verifiers import (
    EATVerifier,
    EntrocutVerifier,
    LLMSufficiencyVerifier,
    PUMAVerifier,
    ProbeLogprobVerifier,
)
from replay_eval.v2.config.schema import (
    ModelConfig,
    ReplayV2Config,
    RuntimeConfig,
    load_config,
)
from replay_eval.v2.engine.cache import RequestCache
from replay_eval.v2.engine.access import EngineAccessGate
from replay_eval.v2.engine.extraction import (
    ExtractionResult,
    HiddenStateTrace,
    PrecomputedExtractionConnector,
)
from replay_eval.v2.engine.types import (
    RequestKind,
    RequestResult,
    RequestSpec,
    Timing,
)
from replay_eval.v2.rendering.hf import CanonicalRenderer, split_reasoning
from replay_eval.v2.runtime.orchestrator import AsyncReplayOrchestrator
from replay_eval.v2.runtime.metrics import RunMetrics
from replay_eval.v2.runtime.runner import (
    ReplayV2Runner,
    _discard_previous_replay_logs,
    _completed_method_names,
    _merge_method_results,
    _missing_dataset_record_numbers,
    _missing_record_numbers,
    _read_jsonl,
    _recover_missing_records,
    _select_method_specs,
)
from replay_eval.v2.runtime.hidden_models import (
    TerminatorHiddenStateVerifier,
    ThoughtCalibrationVerifier,
    _load_terminator_probe,
    build_terminator_method,
    build_thought_calibration_method,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]

    @staticmethod
    def decode(ids, **kwargs):
        return "".join(chr(item) for item in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt=False,
        continue_final_message=False,
        **kwargs,
    ):
        self.calls.append((messages, continue_final_message))
        source = (
            "".join(
                f"{item['role']}:{item.get('content', '')}|"
                for item in messages
                if item["role"] != "assistant"
            )
            + "assistant:"
        )
        assistants = [item for item in messages if item["role"] == "assistant"]
        if assistants:
            assistant = assistants[-1]
            output = assistant.get("thinking", "")
            if "thinking" in assistant and assistant.get("content"):
                output += "</think>" + assistant["content"]
            elif "thinking" not in assistant:
                output = assistant.get("content", "")
            rendered = source + output
        else:
            rendered = (
                source if add_generation_prompt else source.removesuffix("assistant:")
            )
        return self.encode(rendered) if tokenize else rendered


class FakeEngine:
    def __init__(self, tokenizer, probe_probability=0.99):
        self.tokenizer = tokenizer
        self.probe_probability = probe_probability
        self.calls = []

    async def submit(self, spec):
        self.calls.append(spec)
        if spec.kind == RequestKind.NORMAL_PROBE:
            text = "{42}"
            probability = self.probe_probability
        elif spec.kind == RequestKind.DTSR:
            text = "Confidence: \\boxed{90}"
            probability = 0.9
        elif spec.kind == RequestKind.FINAL:
            text = "42"
            probability = 0.9
        else:
            text = "r</think>42"
            probability = 0.9
        rows = [
            LogProb(
                token=char,
                logprob=__import__("math").log(probability),
                top_logprobs=[
                    TopLogProb(token=char, logprob=__import__("math").log(probability))
                ],
            )
            for char in text
        ]
        return RequestResult(
            response=Response(
                content=text,
                logprobs=rows,
                prompt_tokens=len(spec.prompt_token_ids),
                completion_tokens=len(rows),
            ),
            prompt_token_ids=list(spec.prompt_token_ids),
            output_token_ids=self.tokenizer.encode(text),
        )


class LifecycleEngine(FakeEngine):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.started_count = 0
        self.shutdown_count = 0
        self.load_seconds = 0.25

    async def start(self):
        self.started_count += 1

    async def shutdown(self):
        self.shutdown_count += 1

    @staticmethod
    def peak_gpu_memory_bytes():
        return 123


def response_for(content: str, prompt: str) -> Response:
    return Response(
        input=prompt,
        content=content,
        logprobs=[LogProb(token=char, logprob=-0.1) for char in content],
        completion_tokens=len(content),
    )


class V2Tests(unittest.TestCase):
    def test_continued_replay_and_result_groups(self):
        async def exercise():
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = renderer.source_text(messages)

            def verify(ctx):
                ctx.logs.setdefault("positions", []).append(ctx.position)
                return ctx.position in (2, 4)

            orchestrator = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer), renderer=renderer, exit=False,
                methods=[EarlyExitMethod("deer", AlwaysCheckpoint(), verify,
                                         lambda value, ctx: value)],
            )
            _, results, _ = await orchestrator.evaluate(
                example_id="continued", question="q", messages=messages,
                trajectory=response_for("abcd</think>full", prompt),
            )
            rows = [r.to_dict() for r in results]
            self.assertEqual([r["method"] for r in rows],
                             ["deer:1", "deer:2", "deer:full"])
            self.assertEqual(rows[0]["logs"]["positions"], [1, 2])
            self.assertEqual(rows[-1]["final_answer"], "full")
            self.assertEqual(_completed_method_names(rows, exit=False), {"deer"})
            self.assertEqual(_completed_method_names(rows[:-1], exit=False), set())
            self.assertEqual(_completed_method_names(rows, exit=True), set())
            self.assertEqual(_completed_method_names([{"method": "deer"}], exit=False), set())
            old = rows + [{"method": "deer:3", "base_method": "deer", "exit": False},
                          {"method": "other"}]
            self.assertEqual(_merge_method_results(old, rows), [{"method": "other"}] + rows)
            # Final-answer reruns retain the base method lookup and sample identity.
            await orchestrator.rerun_final_answers(
                example_id="continued", messages=messages,
                trajectory=response_for("abcd</think>full", prompt), results=rows,
            )
            self.assertEqual(rows[0]["base_method"], "deer")

        asyncio.run(exercise())

    def test_exit_config_and_cli(self):
        import sys
        from scripts import runner
        from replay_eval.v2 import __main__ as v2_main
        import yaml

        for flag in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "config.yml"
                config_path.write_text(yaml.safe_dump({
                    "llm": {"models": {"fake": {"id": "vllm:fake"}}},
                    "agent_kwargs": {"replay_eval": {"exit": flag, "v2": {}}},
                }))
                self.assertEqual(load_config(config_path, model_name="fake").runtime.exit, flag)
            with patch.object(sys, "argv", ["runner", "--exit", str(flag)]):
                self.assertEqual(runner.parse_args().exit, flag)
            with patch.object(sys, "argv", ["v2", "--model", "fake",
                              "--input-file", "in", "--output-file", "out",
                              f"--exit={str(flag).lower()}"]):
                self.assertEqual(v2_main.parse_args().exit, flag)

    def test_shared_runner_applies_exit_override_after_reloading_v2_config(self):
        import sys
        from scripts import runner

        config = ReplayV2Config(model=ModelConfig(id="fake"))
        with patch.object(sys, "argv", ["runner", "--model", "fake",
                          "--input-file", "in", "--exit", "false"]):
            args = runner.parse_args()
        with patch("replay_eval.v2.config.schema.load_config", return_value=config), \
             patch.object(runner, "setup_logging"), \
             patch.object(runner, "load_optional_hf_token"), \
             patch.object(config, "validate", side_effect=RuntimeError("validated")):
            with self.assertRaisesRegex(RuntimeError, "validated"):
                runner.run_v2(args, {})
        self.assertFalse(config.runtime.exit)

    def test_exit_flag_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, "exit.*boolean"):
            ReplayV2Config(model=ModelConfig(id="fake"), runtime=RuntimeConfig(exit="false"))

    def test_qwen3_replay_preserves_inline_thinking_when_template_strips_it(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel(
            {word: index for index, word in enumerate([
                "[UNK]", "user", "q", "assistant", "<think>", "reason",
                "</think>", "answer",
            ])},
            unk_token="[UNK]",
        ))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        # Reproduce Qwen3's removal of inline reasoning from assistant content
        # using the real Transformers continuation validation.
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] + ' ' }}"
            "{% if message['role'] == 'assistant' %}"
            "{{ message['content'].split('</think>')[-1].strip() + ' ' }}"
            "{% else %}{{ message['content'] + ' ' }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
        )
        messages = [{"role": "user", "content": "q"}]
        content = "<think> reason </think> answer"
        with self.assertRaisesRegex(ValueError, "final message"):
            tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": content}],
                continue_final_message=True,
            )
        renderer = CanonicalRenderer(
            tokenizer, model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        )
        source_text = renderer.source_text(messages)
        source_ids = renderer.source(messages)
        for output_ids in (None, tokenizer.encode(content, add_special_tokens=False)):
            with self.subTest(output_ids=output_ids):
                trace = renderer.full_trace(
                    messages, content, stored_input=source_text, output_token_ids=output_ids,
                )
                expected_output = tokenizer.encode(content, add_special_tokens=False)
                self.assertEqual(list(trace.full_token_ids), source_ids + expected_output)
                self.assertEqual(list(trace.output_token_ids), expected_output)
                self.assertEqual(trace.thinking_token_count, 2)
        for prefix in ("", "<think> reason", content):
            self.assertEqual(
                renderer.checkpoint(messages, prefix, source_token_ids=source_ids),
                source_ids + tokenizer.encode(prefix, add_special_tokens=False),
            )
        # An explicitly supplied prompt prefix remains authoritative.
        self.assertEqual(
            renderer.checkpoint(messages, "reason", source_token_ids=[123]),
            [123, *tokenizer.encode("reason", add_special_tokens=False)],
        )

    def test_gemma4_uses_reasoning_field_and_disables_it_for_baseline(self):
        class Gemma4Tokenizer(FakeTokenizer):
            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt=False,
                continue_final_message=False,
                **kwargs,
            ):
                self.calls.append((messages, continue_final_message))
                source = "".join(
                    f"{item['role']}:{item.get('content', '')}|"
                    for item in messages
                    if item["role"] != "assistant"
                ) + "assistant:"
                assistants = [item for item in messages if item["role"] == "assistant"]
                if assistants:
                    assistant = assistants[-1]
                    if "reasoning" in assistant:
                        source += f"<thought>{assistant['reasoning']}"
                        if continue_final_message != "reasoning":
                            source += f"</thought>{assistant.get('content', '')}"
                    else:
                        source += assistant.get("content", "")
                return self.encode(source) if tokenize else source

        tokenizer = Gemma4Tokenizer()
        renderer = CanonicalRenderer(
            tokenizer,
            model_id="google/gemma-4-E4B-it",
            think_start_token="<thought>",
            think_end_token="</thought>",
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "q"},
        ]

        thinking_source = tokenizer.decode(renderer.source(messages))
        self.assertIn("system:system <|think|>", thinking_source)
        renderer.checkpoint(messages, "reason")
        assistant = tokenizer.calls[-1][0][-1]
        self.assertEqual(
            assistant,
            {"role": "assistant", "content": "", "reasoning": "reason"},
        )
        self.assertEqual(tokenizer.calls[-1][1], "reasoning")

        baseline_source = tokenizer.decode(
            renderer.source(messages, enable_thinking=False)
        )
        self.assertNotIn("<|think|>", baseline_source)
        baseline_assistant = renderer._assistant("", "answer", finished=True)
        self.assertEqual(
            baseline_assistant,
            {"role": "assistant", "content": "answer"},
        )
        self.assertNotIn("reasoning", baseline_assistant)

    def test_gemma4_no_thinking_baseline_uses_fresh_direct_answer_prompt(self):
        async def exercise(exit):
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(
                tokenizer,
                model_id="google/gemma-4-E4B-it",
                think_start_token="<thought>",
                think_end_token="</thought>",
            )
            method = EarlyExitMethod(
                "baseline_no_thinking",
                AlwaysCheckpoint(),
                None,
                None,
                no_thinking=True,
            )
            engine = FakeEngine(tokenizer)
            orchestrator = AsyncReplayOrchestrator(
                engine=engine,
                renderer=renderer,
                methods=[method],
                exit=exit,
                completion_kwargs={"temperature": 0.0},
            )
            messages = [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "q"},
            ]
            prompt = renderer.source_text(messages)
            trajectory = response_for("reason</thought>answer", prompt)

            _trajectory, results, _timing = await orchestrator.evaluate(
                example_id="gemma-baseline",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )

            self.assertEqual(results[0].position, 0)
            self.assertEqual(results[0].thinking_tokens, 0)
            self.assertEqual(results[0].reasoning_prefix, "")
            final_spec = next(call for call in engine.calls if call.kind == RequestKind.FINAL)
            final_prompt = tokenizer.decode(final_spec.prompt_token_ids)
            self.assertNotIn("<|think|>", final_prompt)
            self.assertNotIn("reason", final_prompt)
            self.assertEqual(final_prompt, "system:system|user:q|assistant:")

        for exit in (True, False):
            asyncio.run(exercise(exit))

    def test_gpt_oss_reasoning_split_uses_last_end_and_removes_partial_end(self):
        self.assertEqual(
            split_reasoning("abc</thi", "</think>", "<think>"),
            ("abc", "", False),
        )
        self.assertEqual(
            split_reasoning(
                "first</think>second</think>answer",
                "</think>",
                "<think>",
            ),
            ("first</think>second", "answer", True),
        )

    def test_non_gpt_oss_reasoning_split_keeps_original_boundary_behavior(self):
        self.assertEqual(
            split_reasoning("abc</thi", "</think>"),
            ("abc</thi", "", False),
        )
        self.assertEqual(
            split_reasoning("first</think>second</think>answer", "</think>"),
            ("first", "second</think>answer", True),
        )

    def test_checkpoint_truncates_before_last_rendered_thinking_end(self):
        class ClosingTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                rendered = super().apply_chat_template(messages, **kwargs)
                if kwargs.get("tokenize") and any(
                    message.get("role") == "assistant" for message in messages
                ):
                    rendered.extend(self.encode("</think>middle</think>tail"))
                return rendered

        tokenizer = ClosingTokenizer()
        renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
        messages = [{"role": "user", "content": "q"}]
        rendered = renderer.checkpoint(messages, "abc")
        self.assertEqual(
            tokenizer.decode(rendered),
            "user:q|assistant:abc</think>middle",
        )

    def test_gpt_oss_realigns_token_boundary_without_changing_source_prompt(self):
        class BoundaryMismatchTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                rendered = super().apply_chat_template(messages, **kwargs)
                if kwargs.get("tokenize") and any(
                    message.get("role") == "assistant" for message in messages
                ):
                    rendered[len(self.encode("user:q|assistant:")) - 1] = ord("?")
                return rendered

        tokenizer = BoundaryMismatchTokenizer()
        renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
        messages = [{"role": "user", "content": "q"}]
        source_ids = renderer.source(messages)
        trace = renderer.full_trace(messages, "abc")

        self.assertEqual(list(trace.full_token_ids[: len(source_ids)]), source_ids)
        self.assertEqual(tokenizer.decode(trace.output_token_ids), "abc")

    def test_gpt_oss_replay_preserves_stored_prompt_date(self):
        class DatedTokenizer(FakeTokenizer):
            current_date = "2026-09-13"

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt=False,
                continue_final_message=False,
                **kwargs,
            ):
                prompt = (
                    "<|start|>system<|message|>Knowledge cutoff: 2024-06\n"
                    f"Current date: {self.current_date}\n<|end|>"
                    f"<|start|>user<|message|>{messages[0]['content']}<|end|>"
                    "<|start|>assistant<|channel|>analysis<|message|>"
                )
                assistants = [
                    item for item in messages if item["role"] == "assistant"
                ]
                if assistants:
                    prompt += assistants[-1].get("thinking", "")
                return self.encode(prompt) if tokenize else prompt

        tokenizer = DatedTokenizer()
        renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
        messages = [{"role": "user", "content": "q"}]
        tokenizer.current_date = "2026-09-12"
        stored_input = renderer.source_text(messages)
        stored_ids = tokenizer.encode(stored_input)
        tokenizer.current_date = "2026-09-13"

        trace = renderer.full_trace(
            messages,
            "abc",
            stored_input=stored_input,
            output_token_ids=tokenizer.encode("abc"),
        )

        self.assertEqual(list(trace.source_token_ids), stored_ids)
        self.assertEqual(
            list(trace.full_token_ids), [*stored_ids, *tokenizer.encode("abc")]
        )

    def test_gpt_oss_replay_rejects_non_date_prompt_changes(self):
        class DatedTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                prompt = (
                    "<|start|>system<|message|>Knowledge cutoff: 2024-06\n"
                    "Current date: 2026-09-13\n<|end|>"
                    f"<|start|>user<|message|>{messages[0]['content']}<|end|>"
                    "<|start|>assistant<|channel|>analysis<|message|>"
                )
                return self.encode(prompt) if tokenize else prompt

        tokenizer = DatedTokenizer()
        renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
        messages = [{"role": "user", "content": "q"}]
        stored_input = renderer.source_text(messages).replace(
            "<|message|>q<|end|>", "<|message|>other<|end|>"
        )

        with self.assertRaisesRegex(ValueError, "not the canonical HF-rendered prompt"):
            renderer.validate_response_input(messages, stored_input)

    def test_indexed_checkpoint_materializes_exact_prefix_only_when_fired(self):
        async def exercise():
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("abcd</think>42", prompt)
            seen = []

            def verifier(ctx):
                seen.append(
                    (
                        ctx.position,
                        ctx.prefix.content,
                        len(ctx.prefix.logprobs),
                        ctx.prefix.get_generation_content(),
                    )
                )
                return float(ctx.position >= 4)

            method = EarlyExitMethod(
                "indexed",
                EveryNTokens(2),
                verifier,
                ThresholdGE(1.0),
            )
            orchestrator = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer),
                renderer=renderer,
                methods=[method],
                completion_kwargs={"temperature": 0.0},
            )
            _response, results, _timing = await orchestrator.evaluate(
                example_id="indexed",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )
            self.assertEqual(
                seen,
                [
                    (2, "ab", 2, prompt + "ab"),
                    (4, "abcd", 4, prompt + "abcd"),
                ],
            )
            self.assertTrue(results[0].stopped_early)
            self.assertEqual(results[0].position, 4)

        asyncio.run(exercise())

    def test_custom_checkpoint_keeps_full_prefix_fallback(self):
        async def exercise():
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("abc</think>42", prompt)
            seen = []

            def checkpoint(ctx):
                seen.append((ctx.prefix.content, len(ctx.prefix.logprobs)))
                return ctx.position == 2

            method = EarlyExitMethod("custom", checkpoint, None, None)
            orchestrator = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer),
                renderer=renderer,
                methods=[method],
                completion_kwargs={"temperature": 0.0},
            )
            _response, results, _timing = await orchestrator.evaluate(
                example_id="custom",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )
            self.assertEqual(seen, [("a", 1), ("ab", 2)])
            self.assertEqual(results[0].position, 2)

        asyncio.run(exercise())

    def test_min_interval_accepts_first_checkpoint_and_uses_last_accepted_position(self):
        async def replay(candidate_positions):
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("x" * 300 + "</think>42", prompt)
            checked = []

            method = EarlyExitMethod(
                "interval",
                lambda ctx: ctx.position in candidate_positions,
                lambda ctx: checked.append(ctx.position) or 0.0,
                ThresholdGE(1.0),
            )
            orchestrator = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer),
                renderer=renderer,
                methods=[method],
                completion_kwargs={"temperature": 0.0},
                min_interval=128,
            )
            await orchestrator.evaluate(
                example_id="interval",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )
            return checked

        self.assertEqual(asyncio.run(replay({1, 63, 129, 130, 257})), [1, 129, 257])
        self.assertEqual(asyncio.run(replay({1, 63, 130, 257, 258})), [1, 130, 258])

    def test_hidden_capture_gate_is_exclusive_and_writer_preferred(self):
        async def exercise():
            gate = EngineAccessGate()
            first_reader_started = asyncio.Event()
            release_first_reader = asyncio.Event()
            order = []

            async def first_reader():
                async with gate.shared():
                    order.append("reader-1-start")
                    first_reader_started.set()
                    await release_first_reader.wait()
                    order.append("reader-1-end")

            async def writer():
                await first_reader_started.wait()
                async with gate.exclusive():
                    order.append("capture")

            async def second_reader():
                await first_reader_started.wait()
                await asyncio.sleep(0)
                async with gate.shared():
                    order.append("reader-2")

            first = asyncio.create_task(first_reader())
            capture = asyncio.create_task(writer())
            await asyncio.sleep(0)
            second = asyncio.create_task(second_reader())
            await asyncio.sleep(0)
            release_first_reader.set()
            await asyncio.gather(first, capture, second)
            self.assertEqual(
                order,
                ["reader-1-start", "reader-1-end", "capture", "reader-2"],
            )

        asyncio.run(exercise())

    def test_llm_call_metrics_exclude_cache_hits(self):
        metrics = RunMetrics()
        response = Response(content="x", completion_tokens=1)
        fresh = RequestResult(
            response=response,
            prompt_token_ids=[1],
            output_token_ids=[2],
            timing=Timing(end_to_end=2.5),
        )
        cached = RequestResult(
            response=response,
            prompt_token_ids=[1],
            output_token_ids=[2],
            timing=Timing(end_to_end=2.5),
            cached=True,
        )
        metrics.add_request(RequestKind.NORMAL_PROBE, fresh)
        metrics.add_request(RequestKind.NORMAL_PROBE, cached)
        summary = metrics.summary()
        self.assertEqual(summary["llm_calls"], 1)
        self.assertEqual(summary["llm_call_seconds"], 2.5)
        self.assertEqual(summary["avg_llm_call_seconds"], 2.5)

    def test_main_config_enables_terminator_16_by_10_scan(self):
        config_path = Path(__file__).resolve().parents[3] / "scripts" / "config.yml"
        config = load_config(config_path, model_name="qwen35_4b")
        terminator = next(
            item
            for item in config.methods
            if item.get("kwargs", {}).get("name") == "terminator_hf"
        )
        kwargs = terminator["kwargs"]
        self.assertNotIn("check_every", kwargs)
        self.assertNotIn("checkpoint", kwargs)
        self.assertNotIn("base_model", kwargs)
        self.assertEqual(
            kwargs["artifact_dir"],
            "checkpoints/terminator/replay/qwen35_4b",
        )
        self.assertEqual(kwargs["replay_tokens"], 16)
        self.assertEqual(kwargs["window_size"], 10)

        self.assertTrue(config.runtime.rerun_all)
        self.assertEqual(config.runtime.min_interval, 128)
        self.assertEqual(config.runtime.example_concurrency, 4)
        self.assertEqual(config.runtime.example_timeout_seconds, 300)
        thought = next(
            item
            for item in config.methods
            if item.get("kwargs", {}).get("name") == "thought_calibration"
        )
        self.assertEqual(thought["kwargs"]["risk_tolerance"], 0.05)
        self.assertEqual(thought["kwargs"]["check_every"], 64)
        self.assertEqual(config.model.id, "Qwen/Qwen3.5-4B")
        self.assertEqual(config.model.key, "qwen35_4b")
        self.assertEqual(config.model.dtype, "bfloat16")
        self.assertTrue(config.model.engine_kwargs["enable_prefix_caching"])
        self.assertTrue(config.model.engine_kwargs["enable_chunked_prefill"])
        self.assertEqual(config.model.engine_kwargs["max_num_seqs"], 4)
        self.assertEqual(config.model.engine_kwargs["max_num_batched_tokens"], 2048)
        self.assertTrue(config.model.engine_kwargs["enforce_eager"])
        self.assertEqual(
            config.model.engine_kwargs["attention_config"],
            {"backend": "TRITON_ATTN"},
        )
        self.assertEqual(config.model.engine_kwargs["gdn_prefill_backend"], "triton")
        self.assertEqual(
            config.model.environment["VLLM_USE_FLASHINFER_SAMPLER"], "0"
        )
        self.assertNotIn("speculative_config", config.model.engine_kwargs)
        qwen_9b = load_config(config_path, model_name="qwen35_9b")
        self.assertEqual(
            qwen_9b.model.engine_kwargs["attention_config"],
            {"backend": "TRITON_ATTN"},
        )
        self.assertEqual(
            qwen_9b.model.engine_kwargs["gdn_prefill_backend"], "triton"
        )
        self.assertNotIn("speculative_config", qwen_9b.model.engine_kwargs)
        self.assertTrue(config.rendering.template_kwargs["enable_thinking"])
        self.assertEqual(config.runtime.code_probe_token_multiplier, 1)
        self.assertEqual(config.runtime.code_threshold_multiplier, 1.0)
        self.assertEqual(config.runtime.max_temp_answer_completion_token, 16)
        self.assertEqual(config.runtime.max_final_answer_completion_token, 1024)
        self.assertEqual(config.runtime.sampling["max_tokens"], 38000)
        self.assertEqual(config.model.max_model_len, 38000)
        method_names = [item["kwargs"]["name"] for item in config.methods]
        self.assertEqual(len(method_names), 22)
        self.assertEqual(len(set(method_names)), 22)
        self.assertEqual(
            method_names[-2:],
            ["terminator_hf", "thought_calibration"],
        )
        self.assertEqual(config.extraction.dola_layers, [])
        self.assertNotIn("dola_disagreement", method_names)
        self.assertEqual(config.runtime.verify_batch_token, 16)
        puma = next(item for item in config.methods if item["kwargs"]["name"] == "puma")
        self.assertEqual(puma["kwargs"]["embedding_device"], "cuda")
        dtsr = next(item for item in config.methods if item["kwargs"]["name"] == "dtsr")
        self.assertEqual(dtsr["kwargs"]["probe_tokens"], 20)

        processed = set(method_names[:5])
        missing = _select_method_specs(
            config.methods, processed_methods=processed, rerun_all=False
        )
        self.assertEqual([item["kwargs"]["name"] for item in missing], method_names[5:])
        self.assertEqual(
            _select_method_specs(
                config.methods, processed_methods=processed, rerun_all=True
            ),
            config.methods,
        )
        merged = _merge_method_results(
            [{"method_name": method_names[0], "final_answer": "old"}],
            [{"method_name": method_names[0], "final_answer": "new"}],
        )
        self.assertEqual(merged[0]["final_answer"], "new")
        cleaned = _discard_previous_replay_logs(
            {
                "input": {"question": "q"},
                "additional_logs": {"method_results": [{"method": "old"}]},
                "method_results": [{"method": "old"}],
                "v2_timing": {"old": True},
                "response": {
                    "content": "source trajectory",
                    "method_results_dicts": [{"method": "old"}],
                },
            }
        )
        self.assertEqual(cleaned["input"], {"question": "q"})
        self.assertEqual(cleaned["response"], {"content": "source trajectory"})
        self.assertNotIn("additional_logs", cleaned)
        self.assertNotIn("method_results", cleaned)
        self.assertNotIn("v2_timing", cleaned)

        from scripts.config_loader import (
            load_config as load_shared_config,
        )
        from scripts.runner import get_agent_kwargs, get_replay_version

        shared = load_shared_config(config_path)
        self.assertEqual(config.runtime.sampling["temperature"], 0.4)
        self.assertEqual(get_replay_version(shared), 1)
        v1_kwargs = get_agent_kwargs(shared, "replay_eval")
        self.assertNotIn("version", v1_kwargs)
        self.assertNotIn("v2", v1_kwargs)
        shared["agent_kwargs"]["replay_eval"]["version"] = 2
        self.assertEqual(get_replay_version(shared), 2)

    def test_main_gemma4_config_preserves_channel_special_tokens(self):
        config_path = Path(__file__).resolve().parents[3] / "scripts" / "config.yml"
        config = load_config(config_path, model_name="gemma4e4")

        self.assertFalse(config.runtime.sampling["skip_special_tokens"])

    def test_global_vllm_defaults_and_model_specific_key_overrides(self):
        payload = {
            "agent_kwargs": {"replay_eval": {"v2": {}}},
            "llm": {
                "vllm": {
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.8,
                    "environment": {
                        "GLOBAL_ENV": "global",
                        "OVERRIDDEN_ENV": "global",
                    },
                    "engine_kwargs": {
                        "global_engine": True,
                        "overridden_engine": "global",
                    },
                    "serving": {
                        "global_serving": True,
                        "overridden_serving": "global",
                    },
                },
                "models": {
                    "defaults": {"id": "vllm:default-model"},
                    "overrides": {
                        "id": "vllm:override-model",
                        "vllm": {
                            "gpu_memory_utilization": 0.5,
                            "environment": {"OVERRIDDEN_ENV": "model"},
                            "engine_kwargs": {
                                "overridden_engine": "model",
                            },
                            "serving": {
                                "overridden_serving": "model",
                            },
                        },
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            defaults = load_config(config_path, model_name="defaults")
            overrides = load_config(config_path, model_name="overrides")

        self.assertEqual(defaults.model.dtype, "bfloat16")
        self.assertEqual(defaults.model.gpu_memory_utilization, 0.8)
        self.assertEqual(
            defaults.model.environment,
            {"GLOBAL_ENV": "global", "OVERRIDDEN_ENV": "global"},
        )
        self.assertEqual(
            defaults.model.engine_kwargs,
            {
                "global_serving": True,
                "overridden_serving": "global",
                "global_engine": True,
                "overridden_engine": "global",
            },
        )
        self.assertEqual(overrides.model.dtype, "bfloat16")
        self.assertEqual(overrides.model.gpu_memory_utilization, 0.5)
        self.assertEqual(
            overrides.model.environment,
            {"GLOBAL_ENV": "global", "OVERRIDDEN_ENV": "model"},
        )
        self.assertEqual(
            overrides.model.engine_kwargs,
            {
                "global_serving": True,
                "overridden_serving": "model",
                "global_engine": True,
                "overridden_engine": "model",
            },
        )

    def test_config_rejects_duplicate_names_and_external_llms(self):
        duplicate = {
            "factory": "replay_eval.methods.build_baseline_no_thinking",
            "kwargs": {"name": "same"},
        }
        with self.assertRaisesRegex(ValueError, "Duplicate replay method name"):
            ReplayV2Config(
                model=ModelConfig(id="model"), methods=[duplicate, dict(duplicate)]
            )
        custom = {
            "factory": "replay_eval.methods.build_dtsr_method",
            "kwargs": {"name": "custom", "llm": "custom"},
        }
        with self.assertRaisesRegex(ValueError, "unsupported external LLM"):
            ReplayV2Config(model=ModelConfig(id="model"), methods=[custom])
        custom["kwargs"] = {"name": "custom", "llm": "same", "llm_name": "remote"}
        with self.assertRaisesRegex(ValueError, "unsupported external LLM"):
            ReplayV2Config(model=ModelConfig(id="model"), methods=[custom])

    @staticmethod
    def _hidden_context(verifier, trace, *, position, state=None, logs=None):
        class Agent:
            hidden_trace = trace

        return VerifierContext(
            question="q",
            prefix=Response(content="x" * position, completion_tokens=position),
            latest_token="x",
            position=position,
            agent=Agent(),
            kwargs={},
            state=state if state is not None else {},
            logs=logs if logs is not None else {},
        )

    def test_hidden_models_reject_unverified_token_embeddings(self):
        import torch

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def predict_proba(self, hidden_states, **kwargs):
                return torch.ones(hidden_states.shape[:-1])

        verifier = TerminatorHiddenStateVerifier(
            probe=Probe(), expected_model=None, score_key="terminator"
        )
        trace = HiddenStateTrace(
            raw=torch.zeros((2, 3)),
            final_normalized=torch.zeros((2, 3)),
            token_ids=[1, 2],
            prompt_tokens=4,
            hidden_state_contract="vllm_token_embed",
        )
        with self.assertRaisesRegex(RuntimeError, "pre_final_rmsnorm"):
            verifier(self._hidden_context(verifier, trace, position=1))

    def test_terminator_adapter_preserves_absolute_positions_and_worker_path(self):
        import torch

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.position_ids = None

            def predict_proba(self, hidden_states, **kwargs):
                self.position_ids = kwargs["position_ids"].detach().clone()
                return hidden_states[..., 0] / 10

        probe = Probe()
        method = build_terminator_method(
            probe=probe,
            base_model="model",
            threshold=0.5,
            replay_tokens=2,
            window_size=2,
            worker_score_key="terminator",
            position_mode="absolute_prompt",
        )
        trace = HiddenStateTrace(
            raw=torch.tensor([[1.0, 0.0], [8.0, 0.0]]),
            final_normalized=torch.zeros((2, 2)),
            token_ids=[1, 2],
            prompt_tokens=7,
            hidden_state_contract="pre_final_rmsnorm",
            model_id="model",
        )
        ctx = self._hidden_context(method.verifier, trace, position=2)
        score = method.verify(ctx)
        self.assertEqual(score, 1.0)
        self.assertTrue(method.should_stop(score, ctx))
        self.assertEqual(probe.position_ids.tolist(), [[7, 8]])
        trace.release_token_states()
        self.assertIsNone(trace.raw)
        self.assertEqual(method.verify(ctx), score)
        self.assertEqual(ctx.logs["terminator_scans"][-1]["score_source"], "hf_cached")

        worker_trace = HiddenStateTrace(
            raw=torch.empty((2, 0)),
            final_normalized=torch.empty((2, 0)),
            token_ids=[1, 2],
            prompt_tokens=None,
            hidden_state_contract="unknown",
            model_outputs={"terminator": [0.1, 0.9]},
        )
        worker_verifier = TerminatorHiddenStateVerifier(
            probe=None,
            expected_model=None,
            score_key="terminator",
            window_size=1,
        )
        self.assertEqual(
            worker_verifier(
                self._hidden_context(worker_verifier, worker_trace, position=2)
            ),
            1.0,
        )

    def test_terminator_artifact_config_resolves_probe_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / "probe.pt").touch()
            (artifact_dir / "config.json").write_text(
                json.dumps(
                    {
                        "base_model": "model-from-artifact",
                        "hidden_size": 32,
                        "checkpoint_file": "probe.pt",
                        "model_family": "qwen3",
                        "layer_idx": 3,
                        "ffn_num_hidden_layers": 2,
                        "ffn_activation": "tanh",
                        "ffn_intermediate_size": 48,
                        "ffn_dropout": 0.1,
                        "dtype": "bfloat16",
                    }
                ),
                encoding="utf-8",
            )
            sentinel = object()
            with patch(
                "replay_eval.v2.runtime.hidden_models._load_terminator_probe",
                return_value=sentinel,
            ) as load_probe:
                method = build_terminator_method(
                    artifact_dir=str(artifact_dir),
                    worker_score_key="terminator",
                )

            load_probe.assert_called_once_with(
                str(artifact_dir.resolve()), "auto", None
            )
            self.assertIs(method.verifier.probe, sentinel)
            self.assertEqual(
                method.verifier.expected_model, "model-from-artifact"
            )

    def test_terminator_artifact_rejects_model_override_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / "probe.pt").touch()
            (artifact_dir / "config.json").write_text(
                json.dumps(
                    {
                        "base_model": "artifact-model",
                        "hidden_size": 32,
                        "checkpoint_file": "probe.pt",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_terminator_method(
                    artifact_dir=str(artifact_dir),
                    base_model="different-model",
                    probe=object(),
                )

    def test_terminator_loader_uses_artifact_architecture_and_weight(self):
        import torch

        class Config:
            hidden_size = 32
            num_hidden_layers = 32
            layer_types = ["linear_attention"] * 31 + ["full_attention"]

        class FakeProbe:
            instance = None

            def __init__(self, config, **kwargs):
                self.config = config
                self.kwargs = kwargs
                self.loaded = None
                self.to_calls = []
                FakeProbe.instance = self

            def load_checkpoint(self, path, *, map_location):
                self.loaded = (Path(path), map_location)

            def to(self, *args, **kwargs):
                self.to_calls.append((args, kwargs))
                return self

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            weight = artifact_dir / "model-weight.pt"
            weight.touch()
            (artifact_dir / "config.json").write_text(
                json.dumps(
                    {
                        "base_model": "artifact-model",
                        "hidden_size": 32,
                        "checkpoint_file": weight.name,
                        "model_family": "qwen3",
                        "layer_idx": 32,
                        "ffn_num_hidden_layers": 2,
                        "ffn_activation": "tanh",
                        "ffn_intermediate_size": 48,
                        "ffn_dropout": 0.1,
                        "dtype": "bfloat16",
                    }
                ),
                encoding="utf-8",
            )
            _load_terminator_probe.cache_clear()
            with patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=Config(),
            ) as auto_config, patch(
                "terminator.terminator.TerminatorProbe", FakeProbe
            ):
                with self.assertWarnsRegex(UserWarning, "one-based"):
                    loaded = _load_terminator_probe(
                        str(artifact_dir), "cpu", "auto"
                    )

        auto_config.assert_called_once_with("artifact-model")
        self.assertIs(loaded, FakeProbe.instance)
        self.assertEqual(
            FakeProbe.instance.kwargs,
            {
                "model_family": "qwen3",
                "layer_idx": 31,
                "ffn_num_hidden_layers": 2,
                "ffn_activation": "tanh",
                "ffn_intermediate_size": 48,
                "ffn_dropout": 0.1,
            },
        )
        self.assertEqual(FakeProbe.instance.loaded, (weight, "cpu"))
        self.assertIn(((), {"dtype": torch.bfloat16}), FakeProbe.instance.to_calls)
        self.assertIn((("cpu",), {}), FakeProbe.instance.to_calls)

    def test_rollout_exposes_vllm_terminator_scores_at_matching_token(self):
        async def exercise(tmp_path):
            import torch

            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            reasoning = "abcdefghijklmnop"
            trajectory = response_for(reasoning + "</think>42", prompt)
            raw, normalized = tmp_path / "raw.pt", tmp_path / "normalized.pt"
            torch.save(torch.empty((len(trajectory.content), 0)), raw)
            torch.save(torch.empty((len(trajectory.content), 0)), normalized)
            extraction = ExtractionResult(
                token_ids=tokenizer.encode(trajectory.content),
                observed_logprobs=[-0.1] * len(trajectory.content),
                raw_hidden_path=raw,
                normalized_hidden_path=normalized,
                model_outputs={
                    "terminator": [0.1] * 15
                    + [0.9]
                    + [0.0] * (len(trajectory.content) - 16)
                },
            )
            method = build_terminator_method(
                threshold=0.8, worker_score_key="terminator"
            )
            orchestrator = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer),
                renderer=renderer,
                methods=[method],
                extraction_connector=PrecomputedExtractionConnector(
                    {"worker": extraction}
                ),
            )
            _response, results, _timing = await orchestrator.evaluate(
                example_id="worker",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )
            self.assertTrue(results[0].stopped_early)
            self.assertEqual(results[0].position, 16)
            self.assertEqual(results[0].verifier_score, 1.0)
            self.assertFalse(raw.exists())
            self.assertFalse(normalized.exists())

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_terminator_precomputes_once_then_replays_strict_windows(self):
        import torch

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.input_shape = None
                self.position_ids = None
                self.calls = 0

            def predict_proba(self, hidden_states, **kwargs):
                self.calls += 1
                self.input_shape = tuple(hidden_states.shape)
                self.position_ids = kwargs["position_ids"].detach().clone()
                return hidden_states[..., 0]

        probe = Probe()
        probabilities = [0.0] * 32
        probabilities[7:10] = [0.9, 0.9, 0.9]
        trace = HiddenStateTrace(
            raw=torch.tensor(probabilities).unsqueeze(-1),
            final_normalized=torch.zeros((32, 1)),
            token_ids=list(range(32)),
            prompt_tokens=None,
            hidden_state_contract="pre_final_rmsnorm",
            model_id="model",
        )
        method = build_terminator_method(
            threshold=2,
            replay_tokens=16,
            window_size=10,
            classification_threshold=0.5,
            worker_score_key="terminator",
            probe=probe,
            base_model="model",
        )
        logs = {}
        ctx = self._hidden_context(method.verifier, trace, position=32, logs=logs)
        score = method.verify(ctx)
        scan = logs["terminator_scans"][0]
        self.assertEqual(scan["classified_tokens"], 25)
        self.assertEqual(scan["precomputed_tokens"], 32)
        self.assertEqual(scan["score_source"], "hf_precomputed")
        self.assertNotIn("windows", scan)
        self.assertEqual(probe.input_shape, (1, 32, 1))
        self.assertEqual(probe.position_ids.tolist(), [list(range(32))])
        self.assertEqual(probe.calls, 1)
        self.assertEqual(score, 3.0)
        self.assertTrue(method.should_stop(score, ctx))

        probabilities[9] = 0.0
        earlier_ctx = self._hidden_context(
            method.verifier, trace, position=16, logs=logs
        )
        method.verify(earlier_ctx)
        self.assertEqual(probe.calls, 1)
        self.assertEqual(logs["terminator_scans"][1]["score_source"], "hf_cached")

        strict_trace = HiddenStateTrace(
            raw=torch.tensor(probabilities).unsqueeze(-1),
            final_normalized=trace.final_normalized,
            token_ids=trace.token_ids,
            prompt_tokens=None,
            hidden_state_contract="pre_final_rmsnorm",
            model_id="model",
        )
        strict_ctx = self._hidden_context(
            method.verifier, strict_trace, position=32, logs={}
        )
        strict_score = method.verify(strict_ctx)
        self.assertEqual(strict_score, 2.0)
        self.assertFalse(method.should_stop(strict_score, strict_ctx))

    def test_thought_calibration_uses_segment_mean_smoothing_and_artifact_threshold(
        self,
    ):
        import json
        import torch

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, values):
                return values[..., 0] / 20

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            (artifact / "config.json").write_text(
                json.dumps(
                    {
                        "feature_source": "pre_final_rmsnorm",
                        "pooling": "segment_mean",
                        "threshold_comparison": ">=",
                        "thresholds": {"0.05": 0.3},
                        "smoothing_window": 2,
                        "base_model": "model",
                    }
                ),
                encoding="utf-8",
            )
            method = build_thought_calibration_method(
                artifact_dir=str(artifact), probe=Probe(), check_every=2
            )
            self.assertIsInstance(method.verifier, ThoughtCalibrationVerifier)
            trace = HiddenStateTrace(
                raw=torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0], [13.0, 15.0]]),
                final_normalized=torch.zeros((4, 2)),
                token_ids=[1, 2, 3, 4],
                prompt_tokens=9,
                hidden_state_contract="pre_final_rmsnorm",
                model_id="model",
            )
            method.verifier.prepare_hidden_trace(trace)
            self.assertEqual(tuple(trace.pooled_hidden_states[2].shape), (2, 2))
            trace.release_token_states()
            self.assertIsNone(trace.raw)
            state, logs = {}, {}
            first = self._hidden_context(
                method.verifier, trace, position=2, state=state, logs=logs
            )
            first_score = method.verify(first)
            self.assertAlmostEqual(first_score, 0.15)
            self.assertFalse(method.should_stop(first_score, first))
            second = self._hidden_context(
                method.verifier, trace, position=4, state=state, logs=logs
            )
            second_score = method.verify(second)
            self.assertAlmostEqual(second_score, 0.35)
            self.assertTrue(method.should_stop(second_score, second))
            self.assertEqual(logs["thought_calibration_scores"][1]["segment_start"], 2)

    def test_gpt_oss_uses_structured_thinking_and_open_continuation(self):
        class StrictGPTOSSTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                for message in messages:
                    if "<|channel|>" in str(message.get("thinking", "")):
                        raise ValueError("channel tags leaked into thinking")
                return super().apply_chat_template(messages, **kwargs)

        tokenizer = StrictGPTOSSTokenizer()
        think_start = "<|channel|>analysis<|message|>"
        think_end = "<|end|><|start|>assistant<|channel|>final<|message|>"
        renderer = CanonicalRenderer(
            tokenizer,
            model_id="openai/gpt-oss-20b",
            think_start_token=think_start,
            think_end_token=think_end,
        )
        messages = [{"role": "user", "content": "q"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        trace = renderer.full_trace(
            messages,
            f"<|channel|>analysis to=assistant<|message|>reason{think_end}answer",
            stored_input=prompt,
        )
        assistant = tokenizer.calls[-2][0][-1]
        self.assertEqual(
            assistant, {"role": "assistant", "thinking": "reason", "content": "answer"}
        )
        renderer.checkpoint(messages, f"\n {think_start}rea")
        checkpoint_messages, continuation = tokenizer.calls[-1]
        self.assertEqual(
            checkpoint_messages[-1],
            {"role": "assistant", "thinking": "rea", "content": ""},
        )
        self.assertEqual(continuation, "thinking")
        renderer.checkpoint(
            messages,
            f"<|start|>assistant{think_start}reason{think_end}answer",
        )
        finished_messages, continuation = tokenizer.calls[-1]
        self.assertEqual(
            finished_messages[-1],
            {"role": "assistant", "thinking": "reason", "content": "answer"},
        )
        self.assertEqual(continuation, "content")
        self.assertEqual(len(trace.output_token_ids), len("reason</think>answer"))

    def test_request_cache_keys_include_checkpoint_and_probe_kind(self):
        async def exercise():
            cache = RequestCache()
            calls = 0

            async def submit(spec):
                nonlocal calls
                calls += 1
                return RequestResult(
                    Response(content="x"), list(spec.prompt_token_ids), [1]
                )

            base = dict(prompt_token_ids=[1, 2], sampling={"max_tokens": 2})
            normal = RequestSpec.create(
                **base, checkpoint=3, kind=RequestKind.NORMAL_PROBE
            )
            await cache.get_or_submit(normal, submit)
            hit = await cache.get_or_submit(normal, submit)
            await cache.get_or_submit(
                RequestSpec.create(**base, checkpoint=4, kind=RequestKind.NORMAL_PROBE),
                submit,
            )
            await cache.get_or_submit(
                RequestSpec.create(**base, checkpoint=3, kind=RequestKind.DTSR), submit
            )
            await cache.get_or_submit(
                RequestSpec.create(
                    **base,
                    checkpoint=3,
                    kind=RequestKind.NORMAL_PROBE,
                    cache_salt="example-a",
                ),
                submit,
            )
            await cache.get_or_submit(
                RequestSpec.create(
                    **base,
                    checkpoint=3,
                    kind=RequestKind.NORMAL_PROBE,
                    cache_salt="example-b",
                ),
                submit,
            )
            self.assertEqual(calls, 5)
            self.assertTrue(hit.cached)
            self.assertEqual(cache.hits, 1)

        asyncio.run(exercise())

    def test_existing_verifier_and_stop_rule_receive_canned_async_probe(self):
        async def exercise(tmp_path):
            import torch

            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("r</think>42", prompt)
            raw = tmp_path / "raw.pt"
            normalized = tmp_path / "normalized.pt"
            torch.save(torch.zeros((len(trajectory.content), 2)), raw)
            torch.save(torch.zeros((len(trajectory.content), 2)), normalized)
            extraction = ExtractionResult(
                token_ids=tokenizer.encode(trajectory.content),
                observed_logprobs=[-0.1] * len(trajectory.content),
                raw_hidden_path=raw,
                normalized_hidden_path=normalized,
            )
            method = EarlyExitMethod(
                "deer",
                AlwaysCheckpoint(),
                ProbeLogprobVerifier(probe_tokens=2, probe_logprobs=1),
                ThresholdGE(0.95),
            )
            engine = FakeEngine(tokenizer)
            orchestrator = AsyncReplayOrchestrator(
                engine=engine,
                renderer=renderer,
                methods=[method],
                extraction_connector=PrecomputedExtractionConnector({"1": extraction}),
                completion_kwargs={"temperature": 0.0},
            )
            _response, results, _timing = await orchestrator.evaluate(
                example_id="1",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )
            self.assertTrue(results[0].stopped_early)
            self.assertEqual(results[0].position, 1)
            self.assertEqual(results[0].final_answer, "42")
            self.assertIsInstance(method.verifier, ProbeLogprobVerifier)
            self.assertIsInstance(method.stop_rule, ThresholdGE)
            self.assertEqual(
                [call.kind for call in engine.calls],
                [RequestKind.NORMAL_PROBE, RequestKind.FINAL],
            )
            self.assertFalse(raw.exists())
            self.assertFalse(normalized.exists())

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_full_budget_without_answer_channel_forces_final_answer(self):
        async def exercise(tmp_path):
            import torch

            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("unfinished reasoning", prompt)
            raw, normalized = tmp_path / "raw.pt", tmp_path / "normalized.pt"
            torch.save(torch.zeros((len(trajectory.content), 2)), raw)
            torch.save(torch.zeros((len(trajectory.content), 2)), normalized)
            extraction = ExtractionResult(
                token_ids=tokenizer.encode(trajectory.content),
                observed_logprobs=[-0.1] * len(trajectory.content),
                raw_hidden_path=raw,
                normalized_hidden_path=normalized,
            )
            method = EarlyExitMethod(
                "never_stops",
                AlwaysCheckpoint(),
                lambda _ctx: 0.0,
                ThresholdGE(1.0),
            )
            engine = FakeEngine(tokenizer)
            orchestrator = AsyncReplayOrchestrator(
                engine=engine,
                renderer=renderer,
                methods=[method],
                extraction_connector=PrecomputedExtractionConnector(
                    {"answer": extraction}
                ),
                completion_kwargs={"temperature": 0.0, "max_tokens": 38000},
            )
            _response, results, _timing = await orchestrator.evaluate(
                example_id="answer",
                question="q",
                messages=messages,
                trajectory=trajectory,
                kwargs={
                    "max_temp_answer_completion_token": 32,
                    "max_final_answer_completion_token": 2048,
                },
            )
            self.assertFalse(results[0].stopped_early)
            self.assertEqual(results[0].final_answer, "42")
            self.assertEqual(engine.calls[-1].kind, RequestKind.FINAL)
            self.assertEqual(engine.calls[-1].sampling_dict()["max_tokens"], 2048)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_entrocut_configuration_forces_existing_top_50_verifier(self):
        config = ReplayV2Config(
            model=ModelConfig(id="model", max_logprobs=50),
            methods=[
                {
                    "factory": "replay_eval.methods.build_entrocut_method",
                    "kwargs": {"probe_logprobs": 3},
                }
            ],
        )
        self.assertEqual(config.methods[0]["kwargs"]["probe_logprobs"], 50)
        verifier = EntrocutVerifier(probe_logprobs=50)
        self.assertEqual(verifier.probe_logprobs, 50)

    def test_dtsr_is_a_separate_probe_and_uses_original_verifier(self):
        async def exercise(tmp_path):
            import torch

            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("r</think>42", prompt)
            raw, normalized = tmp_path / "raw.pt", tmp_path / "normalized.pt"
            torch.save(torch.zeros((len(trajectory.content), 2)), raw)
            torch.save(torch.zeros((len(trajectory.content), 2)), normalized)
            extraction = ExtractionResult(
                token_ids=tokenizer.encode(trajectory.content),
                observed_logprobs=[-0.1] * len(trajectory.content),
                raw_hidden_path=raw,
                normalized_hidden_path=normalized,
            )
            method = EarlyExitMethod(
                "dtsr",
                AlwaysCheckpoint(),
                LLMSufficiencyVerifier(probe_tokens=17),
                ThresholdGE(80),
            )
            engine = FakeEngine(tokenizer)
            orchestrator = AsyncReplayOrchestrator(
                engine=engine,
                renderer=renderer,
                methods=[method],
                extraction_connector=PrecomputedExtractionConnector({"d": extraction}),
                completion_kwargs={"temperature": 0.0},
            )
            _response, results, timing = await orchestrator.evaluate(
                example_id="d",
                question="q",
                messages=messages,
                trajectory=trajectory,
            )
            self.assertTrue(results[0].stopped_early)
            self.assertEqual(results[0].verifier_score, 90.0)
            self.assertEqual(
                [call.kind for call in engine.calls],
                [RequestKind.DTSR, RequestKind.FINAL],
            )
            self.assertNotEqual(
                engine.calls[0].prompt_token_ids, engine.calls[1].prompt_token_ids
            )
            self.assertEqual(engine.calls[0].sampling_dict()["max_tokens"], 17)
            self.assertEqual(timing["llm_calls"], 2)
            self.assertEqual(timing["avg_llm_call_seconds"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_dtsr_prefers_confidence_label_over_numbers_in_explanation(self):
        content = (
            "The derivation T_eq depends on M_{p2} and K_1, and gives 0.53. "
            "No additional reasoning is required.\n\nConfidence: 100"
        )
        self.assertEqual(LLMSufficiencyVerifier._parse_score(content), 100.0)

    def test_normal_probe_deduplicates_to_largest_top_logprob_need(self):
        async def exercise():
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            agent = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer),
                renderer=renderer,
                methods=[
                    EarlyExitMethod(
                        "deer",
                        AlwaysCheckpoint(),
                        ProbeLogprobVerifier(2, 1),
                        ThresholdGE(2),
                    ),
                    EarlyExitMethod(
                        "entrocut",
                        AlwaysCheckpoint(),
                        EntrocutVerifier(probe_tokens=7),
                        ThresholdGE(2),
                    ),
                ],
                completion_kwargs={"temperature": 0.0},
            )
            phase_kwargs = {
                "max_temp_answer_completion_token": 32,
                "max_final_answer_completion_token": 2048,
            }
            temp_sampling = agent.agent._completion_kwargs_for_phase(
                "temp_answer", phase_kwargs
            )
            final_sampling = agent.agent._completion_kwargs_for_phase(
                "final", phase_kwargs
            )
            self.assertEqual(temp_sampling["max_completion_token"], 32)
            self.assertEqual(final_sampling["max_completion_token"], 2048)
            self.assertNotIn("max_tokens", temp_sampling)
            self.assertNotIn("max_tokens", final_sampling)
            prefix = Response(input="p", content="r", completion_tokens=1)
            fired = []
            for method in agent.methods:
                context = VerifierContext(
                    question="q",
                    prefix=prefix,
                    latest_token="r",
                    position=1,
                    agent=agent.agent,
                    kwargs={},
                    state=method.state,
                    logs=method.logs,
                    trajectory=prefix,
                    method_name=method.name,
                )
                fired.append((method, context))
            await agent._schedule_checkpoint(
                "x", [{"role": "user", "content": "q"}], fired, {}
            )
            calls = agent.engine.calls
            self.assertEqual(len(calls), 1)
            sampling = calls[0].sampling_dict()
            self.assertEqual(sampling["max_tokens"], 7)
            self.assertEqual(sampling["logprobs"], 50)

        asyncio.run(exercise())

    def test_scheduler_handles_puma_buffers_eat_slots_and_prompt_identity(self):
        async def exercise():
            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            engine = FakeEngine(tokenizer)
            puma = PUMAVerifier(probe_tokens=3, probe_logprobs=2)
            deer = ProbeLogprobVerifier(probe_tokens=2, probe_logprobs=1)
            methods = [
                EarlyExitMethod("deer", AlwaysCheckpoint(), deer, ThresholdGE(2)),
                EarlyExitMethod("puma", AlwaysCheckpoint(), puma, ThresholdGE(2)),
            ]
            orchestrator = AsyncReplayOrchestrator(
                engine=engine,
                renderer=renderer,
                methods=methods,
                completion_kwargs={"temperature": 0.0},
                verify_batch_token=4,
            )
            trajectory = response_for("abcdefgh</think>42", "prompt")
            prefix = Response(
                input=trajectory.input,
                content="abc",
                logprobs=trajectory.logprobs[:3],
                completion_tokens=3,
            )
            fired = []
            for method in orchestrator.methods:
                fired.append(
                    (
                        method,
                        VerifierContext(
                            question="q",
                            prefix=prefix,
                            latest_token="c",
                            position=3,
                            agent=orchestrator.agent,
                            kwargs={},
                            state=method.state,
                            logs=method.logs,
                            trajectory=trajectory,
                            method_name=method.name,
                        ),
                    )
                )
            await orchestrator._schedule_checkpoint("puma", messages, fired, {})
            self.assertEqual(sorted(call.checkpoint for call in engine.calls), [3, 4])
            puma_response = puma._generate_trial_answer(fired[1][1], "math")
            self.assertEqual(getattr(puma_response, "_temp_answer_position"), 4)
            self.assertTrue(fired[1][1].logs["puma_trial_prompt_buffers"])
            puma.probe_tokens = 8
            puma_probe = puma._slice_trial_probe(
                response_for("{42}ignored", "trial prompt"), fired[1][1]
            )
            self.assertEqual(puma_probe.content, "{42")
            self.assertEqual(puma_probe.completion_tokens, 3)
            merged_token_probe = puma._slice_trial_probe(
                Response(
                    content="{42}",
                    logprobs=[LogProb(token="{42}", logprob=-0.1)],
                    completion_tokens=1,
                ),
                fired[1][1],
            )
            self.assertEqual(merged_token_probe.content, "{42}")
            self.assertEqual(merged_token_probe.completion_tokens, 1)

            eat_engine = FakeEngine(tokenizer)
            eat = EATVerifier(
                entropy_decoding_steps=2, entropy_rollouts=2, probe_logprobs=3
            )
            shared_methods = [
                EarlyExitMethod("deer", AlwaysCheckpoint(), deer, ThresholdGE(2)),
                EarlyExitMethod("eat", AlwaysCheckpoint(), eat, ThresholdGE(2)),
            ]
            eat_orchestrator = AsyncReplayOrchestrator(
                engine=eat_engine,
                renderer=renderer,
                methods=shared_methods,
                completion_kwargs={"temperature": 0.0},
            )
            eat_fired = []
            for method in eat_orchestrator.methods:
                method.reset()
                eat_fired.append(
                    (
                        method,
                        VerifierContext(
                            question="q",
                            prefix=prefix,
                            latest_token="c",
                            position=3,
                            agent=eat_orchestrator.agent,
                            kwargs={},
                            state=method.state,
                            logs=method.logs,
                            trajectory=trajectory,
                            method_name=method.name,
                        ),
                    )
                )
            await eat_orchestrator._schedule_checkpoint("eat", messages, eat_fired, {})
            self.assertEqual(len(eat_engine.calls), 3)
            self.assertEqual([call.cache_slot for call in eat_engine.calls], [0, 1, 2])
            deer(eat_fired[0][1])
            eat(eat_fired[1][1])

        asyncio.run(exercise())

    def test_cleanup_also_runs_when_verifier_fails(self):
        async def exercise(tmp_path):
            import torch

            tokenizer = FakeTokenizer()
            renderer = CanonicalRenderer(tokenizer, model_id="openai/gpt-oss-20b")
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("r</think>42", prompt)
            raw, normalized = tmp_path / "raw.pt", tmp_path / "normalized.pt"
            torch.save(torch.zeros((len(trajectory.content), 2)), raw)
            torch.save(torch.zeros((len(trajectory.content), 2)), normalized)
            extraction = ExtractionResult(
                token_ids=tokenizer.encode(trajectory.content),
                observed_logprobs=[-0.1] * len(trajectory.content),
                raw_hidden_path=raw,
                normalized_hidden_path=normalized,
            )

            def fail(_ctx):
                raise RuntimeError("boom")

            orchestrator = AsyncReplayOrchestrator(
                engine=FakeEngine(tokenizer),
                renderer=renderer,
                methods=[
                    EarlyExitMethod("failing", AlwaysCheckpoint(), fail, ThresholdGE(1))
                ],
                extraction_connector=PrecomputedExtractionConnector({"f": extraction}),
            )
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await orchestrator.evaluate(
                    example_id="f",
                    question="q",
                    messages=messages,
                    trajectory=trajectory,
                )
            self.assertFalse(raw.exists())
            self.assertFalse(normalized.exists())

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_extraction_alignment_requires_one_logprob_per_token(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            result = ExtractionResult(
                token_ids=[1, 2],
                observed_logprobs=[-1.0],
                raw_hidden_path=tmp_path / "raw.pt",
                normalized_hidden_path=tmp_path / "norm.pt",
            )
            with self.assertRaisesRegex(ValueError, "Every stored output token"):
                result.validate([1, 2])

    def test_rerun_selection_reuses_legacy_output_and_scores_answers(self):
        async def exercise(tmp_path):
            import json
            import torch

            specs = [
                {
                    "factory": "replay_eval.methods.build_baseline_no_thinking",
                    "kwargs": {"name": "already_done"},
                },
                {
                    "factory": "replay_eval.methods.build_baseline_no_thinking",
                    "kwargs": {"name": "missing"},
                },
            ]
            selected = _select_method_specs(
                specs, processed_methods={"already_done"}, rerun_all=False
            )
            self.assertEqual(selected, [specs[1]])
            self.assertEqual(
                _select_method_specs(
                    specs, processed_methods={"already_done"}, rerun_all=True
                ),
                specs,
            )

            tokenizer = FakeTokenizer()
            stored_output = "reason</think>42"
            raw, normalized = tmp_path / "raw.pt", tmp_path / "normalized.pt"
            torch.save(torch.zeros((len(stored_output), 2)), raw)
            torch.save(torch.zeros((len(stored_output), 2)), normalized)
            connector = PrecomputedExtractionConnector(
                {
                    "7": ExtractionResult(
                        token_ids=tokenizer.encode(stored_output),
                        observed_logprobs=[-0.1] * len(stored_output),
                        raw_hidden_path=raw,
                        normalized_hidden_path=normalized,
                    )
                }
            )
            input_path = tmp_path / "legacy.jsonl"
            first_output = tmp_path / "first.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "line_number": 7,
                        "input": {"question": "q", "answer": "42"},
                        "output": stored_output,
                        "logprobs": [],
                        "num_token": len(stored_output),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = ReplayV2Config(
                model=ModelConfig(
                    id="openai/gpt-oss-20b", key="gpt_oss_20b_high"
                ),
                runtime=RuntimeConfig(
                    warmup=False,
                    rerun_all=False,
                    sampling={"temperature": 0.0},
                ),
                methods=[specs[0]],
            )
            engine = LifecycleEngine(tokenizer)
            await ReplayV2Runner(
                config,
                evaluator=lambda prediction, answer: prediction.strip() == answer,
                evaluator_name="exact",
                engine=engine,
                extraction_connector=connector,
            ).run(input_path, first_output)
            row = json.loads(first_output.read_text(encoding="utf-8"))
            self.assertEqual(row["model"], "gpt_oss_20b_high")
            self.assertEqual(row["output"], stored_output)
            self.assertNotIn("logprobs", row)
            self.assertNotIn("logprobs", row["response"])
            self.assertTrue(row["correct"])
            self.assertEqual(row["evaluator"], "exact")
            self.assertEqual(row["method_results"][0]["final_answer"], "42")
            self.assertTrue(row["method_results"][0]["correct"])
            self.assertNotIn(
                RequestKind.FULL_TRACE, [call.kind for call in engine.calls]
            )

            second_output = tmp_path / "second.jsonl"
            second_engine = LifecycleEngine(tokenizer)
            await ReplayV2Runner(
                config, engine=second_engine, extraction_connector=object()
            ).run(first_output, second_output)
            rerun_row = json.loads(second_output.read_text(encoding="utf-8"))
            self.assertEqual(rerun_row["model"], "gpt_oss_20b_high")
            self.assertTrue(rerun_row["v2_timing"]["skipped_methods"])
            self.assertEqual(len(rerun_row["method_results"]), 1)
            self.assertNotIn("logprobs", rerun_row)
            self.assertNotIn("logprobs", rerun_row["response"])
            self.assertEqual(second_engine.calls, [])

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_runner_resumes_existing_output_and_appends_only_pending_rows(self):
        async def exercise(tmp_path):
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps({"id": number, "question": f"q{number}"})
                    for number in (1, 2)
                )
                + "\n",
                encoding="utf-8",
            )
            output_path.write_text(
                json.dumps(
                    {
                        "id": 1,
                        "question": "q1",
                        "method_results": [],
                        "v2_timing": {"example_seconds": 1.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            engine = LifecycleEngine(FakeTokenizer())
            config = ReplayV2Config(
                model=ModelConfig(id="openai/gpt-oss-20b"),
                runtime=RuntimeConfig(warmup=False),
                methods=[],
            )
            metrics = await ReplayV2Runner(
                config,
                engine=engine,
                extraction_connector=object(),
            ).run(input_path, output_path, resume_from=output_path)
            rows = list(_read_jsonl(output_path))
            self.assertEqual([row["id"] for row in rows], [1, 2])
            self.assertEqual(metrics["resumed_examples"], 1)
            self.assertEqual(metrics["examples"], 1)
            self.assertEqual(engine.started_count, 1)
            self.assertEqual(engine.shutdown_count, 1)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_runner_logs_record_failure_and_continues(self):
        async def exercise(tmp_path):
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(
                json.dumps({"id": 1})
                + "\n"
                + json.dumps({"id": 2, "question": "ok"})
                + "\n",
                encoding="utf-8",
            )
            engine = LifecycleEngine(FakeTokenizer())
            config = ReplayV2Config(
                model=ModelConfig(id="openai/gpt-oss-20b"),
                runtime=RuntimeConfig(warmup=False),
                methods=[],
            )
            metrics = await ReplayV2Runner(
                config,
                engine=engine,
                extraction_connector=object(),
            ).run(input_path, output_path)
            rows = list(_read_jsonl(output_path))
            self.assertEqual([row["id"] for row in rows], [2])
            self.assertEqual(metrics["failed_examples"], 1)
            self.assertEqual(metrics["examples"], 1)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_runner_warns_and_skips_record_after_timeout(self):
        async def exercise(tmp_path):
            class TimeoutRunner(ReplayV2Runner):
                async def _evaluate_record(self, index, record):
                    if index == 1:
                        await asyncio.Event().wait()
                    return {
                        "id": record["id"],
                        "_cache_hits": 0,
                        "_cache_misses": 0,
                    }

            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(
                json.dumps({"id": 1, "question": "slow"})
                + "\n"
                + json.dumps({"id": 2, "question": "fast"})
                + "\n",
                encoding="utf-8",
            )
            config = ReplayV2Config(
                model=ModelConfig(id="openai/gpt-oss-20b"),
                runtime=RuntimeConfig(
                    warmup=False,
                    example_concurrency=1,
                    example_timeout_seconds=0.01,
                ),
                methods=[],
            )
            runner = TimeoutRunner(
                config,
                engine=LifecycleEngine(FakeTokenizer()),
                extraction_connector=object(),
            )
            with self.assertLogs(
                "replay_eval.v2.runtime.runner", level="WARNING"
            ) as logs:
                metrics = await runner.run(input_path, output_path)

            rows = list(_read_jsonl(output_path))
            self.assertEqual([row["id"] for row in rows], [2])
            self.assertEqual(metrics["examples"], 1)
            self.assertEqual(metrics["failed_examples"], 1)
            self.assertEqual(metrics["timed_out_examples"], 1)
            self.assertTrue(
                any("skipped record after exceeding" in line for line in logs.output)
            )

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_missing_input_rows_are_recovered_as_fresh_full_method_runs(self):
        records = [
            {"line_number": 1, "question": "existing 1", "method_results": []},
            {"line_number": 2, "question": "existing 2", "method_results": []},
            {"line_number": 4, "question": "existing 4", "method_results": []},
            {"line_number": 11, "question": "existing 11", "method_results": []},
        ]
        dataset = [
            {"question": f"dataset {number}", "answer": str(number)}
            for number in range(1, 12)
        ]

        self.assertEqual(_missing_record_numbers(records), [3, 5, 6, 7, 8, 9, 10])
        self.assertEqual(
            _missing_dataset_record_numbers(records, len(dataset)),
            [3, 5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(_missing_dataset_record_numbers(records, 13)[-2:], [12, 13])
        recovered = _recover_missing_records(
            records, dataset, dataloader_name="gsm8k"
        )

        self.assertEqual(
            [record["line_number"] for record in recovered],
            [3, 5, 6, 7, 8, 9, 10],
        )
        self.assertEqual(recovered[0]["question"], "dataset 3")
        self.assertEqual(recovered[-1]["answer"], "10")
        self.assertEqual(recovered[0]["dataloader"], "gsm8k")
        self.assertNotIn("method_results", recovered[0])

        specs = [
            {
                "factory": "replay_eval.methods.build_baseline_no_thinking",
                "kwargs": {"name": "one"},
            },
            {
                "factory": "replay_eval.methods.build_baseline_no_thinking",
                "kwargs": {"name": "two"},
            },
        ]
        selected = _select_method_specs(
            specs,
            processed_methods=set(),
            rerun_all=False,
        )
        self.assertEqual(selected, specs)

    def test_runner_loads_missing_rows_from_recovery_dataloader(self):
        async def exercise(tmp_path):
            class RecoveryRunner(ReplayV2Runner):
                evaluated = []

                async def _evaluate_record(self, index, record):
                    self.evaluated.append(record)
                    return await super()._evaluate_record(index, record)

            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(
                "\n".join(
                    json.dumps({"line_number": number, "question": "existing"})
                    for number in (1, 3)
                )
                + "\n",
                encoding="utf-8",
            )
            loader_calls = []

            def load_dataset(*, split, limit):
                loader_calls.append((split, limit))
                return [
                    {"question": f"dataset {number}", "answer": str(number)}
                    for number in range(1, 6)
                ]

            runner = RecoveryRunner(
                ReplayV2Config(
                    model=ModelConfig(
                        id="openai/gpt-oss-20b", key="gpt_oss_20b_high"
                    ),
                    runtime=RuntimeConfig(warmup=False),
                ),
                engine=LifecycleEngine(FakeTokenizer()),
                extraction_connector=object(),
            )
            await runner.run(
                input_path,
                output_path,
                recovery_dataloader=load_dataset,
                dataloader_name="gsm8k",
            )

            self.assertEqual(loader_calls, [("test", None)])
            recovered = next(
                record for record in runner.evaluated if record["line_number"] == 2
            )
            self.assertEqual(recovered["question"], "dataset 2")
            self.assertEqual(recovered["dataloader"], "gsm8k")
            self.assertEqual(
                sorted(record["line_number"] for record in runner.evaluated),
                [1, 2, 3, 4, 5],
            )
            output_rows = list(_read_jsonl(output_path))
            self.assertEqual(
                {row["model"] for row in output_rows}, {"gpt_oss_20b_high"}
            )

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_runner_can_rerun_only_a_removed_final_answer(self):
        async def exercise(tmp_path):
            import json

            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "line_number": 1,
                        "input": {"question": "q", "answer": "42"},
                        "output": "reason</think>old",
                        "method_results": [
                            {
                                "method": "already_done",
                                "position": 3,
                                "stopped_early": True,
                                "verifier_score": 0.9,
                                "state": {"kept": True},
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = ReplayV2Config(
                model=ModelConfig(id="openai/gpt-oss-20b"),
                runtime=RuntimeConfig(
                    warmup=False,
                    rerun_all=False,
                    rerun_final_answer=True,
                    sampling={"temperature": 0.0},
                ),
                methods=[
                    {
                        "factory": "replay_eval.methods.build_baseline_no_thinking",
                        "kwargs": {"name": "already_done"},
                    }
                ],
            )
            engine = LifecycleEngine(FakeTokenizer())
            await ReplayV2Runner(
                config,
                evaluator=lambda prediction, answer: prediction.strip() == answer,
                engine=engine,
                extraction_connector=object(),
            ).run(input_path, output_path)
            row = json.loads(output_path.read_text(encoding="utf-8"))
            result = row["method_results"][0]
            self.assertEqual(result["final_answer"], "42")
            self.assertTrue(result["correct"])
            self.assertEqual(result["state"], {"kept": True})
            self.assertTrue(row["v2_timing"]["final_answer_only"])
            self.assertEqual([call.kind for call in engine.calls], [RequestKind.FINAL])

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_runner_flushes_each_row_in_completion_order(self):
        async def exercise(tmp_path):
            import json

            tokenizer = FakeTokenizer()
            second_evaluated = asyncio.Event()
            release_first = asyncio.Event()

            class StreamingRunner(ReplayV2Runner):
                async def _evaluate_record(self, index, record):
                    if index == 1:
                        await release_first.wait()
                    else:
                        second_evaluated.set()
                    return {
                        "id": record["id"],
                        "_cache_hits": index,
                        "_cache_misses": index + 1,
                    }

            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "output.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps({"id": value, "question": "q"}) + "\n"
                    for value in ("one", "two")
                ),
                encoding="utf-8",
            )
            config = ReplayV2Config(
                model=ModelConfig(id="openai/gpt-oss-20b"),
                runtime=RuntimeConfig(
                    warmup=False,
                    example_concurrency=2,
                    sampling={"temperature": 0.0},
                ),
            )
            runner = StreamingRunner(
                config,
                engine=LifecycleEngine(tokenizer),
                extraction_connector=object(),
            )
            task = asyncio.create_task(runner.run(input_path, output_path))
            await second_evaluated.wait()
            await asyncio.sleep(0.01)
            self.assertTrue(output_path.exists())
            first_rows = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(row)["id"] for row in first_rows], ["two"])
            release_first.set()
            metrics = await task
            rows = [
                json.loads(row)
                for row in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["id"] for row in rows], ["two", "one"])
            self.assertEqual(metrics["cache_hits"], 3)
            self.assertEqual(metrics["cache_misses"], 5)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))

    def test_runner_keeps_one_engine_and_writes_ordered_output(self):
        async def exercise(tmp_path):
            import json
            import torch

            tokenizer = FakeTokenizer()
            messages = [{"role": "user", "content": "q"}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            trajectory = response_for("r</think>42", prompt)
            raw, normalized = tmp_path / "raw.pt", tmp_path / "normalized.pt"
            torch.save(torch.zeros((len(trajectory.content), 2)), raw)
            torch.save(torch.zeros((len(trajectory.content), 2)), normalized)
            connector = PrecomputedExtractionConnector(
                {
                    "one": ExtractionResult(
                        token_ids=tokenizer.encode(trajectory.content),
                        observed_logprobs=[-0.1] * len(trajectory.content),
                        raw_hidden_path=raw,
                        normalized_hidden_path=normalized,
                    )
                }
            )
            input_path, output_path = (
                tmp_path / "input.jsonl",
                tmp_path / "output.jsonl",
            )
            input_path.write_text(
                json.dumps(
                    {
                        "id": "one",
                        "question": "q",
                        "response": trajectory.model_dump(),
                        "output_token_ids": tokenizer.encode(trajectory.content),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = ReplayV2Config(
                model=ModelConfig(id="openai/gpt-oss-20b"),
                runtime=RuntimeConfig(warmup=False, sampling={"temperature": 0.0}),
                methods=[{"factory": "replay_eval.methods.build_baseline_no_thinking"}],
            )
            engine = LifecycleEngine(tokenizer)
            metrics = await ReplayV2Runner(
                config, engine=engine, extraction_connector=connector
            ).run(input_path, output_path)
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(engine.started_count, 1)
            self.assertEqual(engine.shutdown_count, 1)
            self.assertEqual(metrics["model_load_seconds"], 0.25)
            self.assertEqual(metrics["warmup_seconds"], 0.0)
            self.assertEqual(row["id"], "one")
            self.assertNotIn("_cache_hits", row)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(Path(directory)))


if __name__ == "__main__":
    unittest.main()
