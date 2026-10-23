from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from methods.response import Response
from replay_eval.checkpoints import AlwaysCheckpoint, EveryNTokens
from replay_eval.method import EarlyExitMethod
from replay_eval.v2.engine.types import RequestKind, RequestResult
from replay_eval.v2.runtime.random_exit_evaluation import (
    AdvisoryThresholds,
    RandomExitReplayEvaluation,
    SampledExitEstimator,
    SamplingSettings,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class Renderer:
    tokenizer = Tokenizer()
    think_end_token = "</think>"
    think_start_token = "<think>"
    is_gpt_oss = False


class Engine:
    def __init__(self):
        self.calls = []

    async def submit(self, spec):
        self.calls.append(spec)
        answer = "yes" if spec.kind == RequestKind.RANDOM_EXIT else "{yes}"
        return RequestResult(Response(content=answer, completion_tokens=len(answer)), list(spec.prompt_token_ids), [ord(c) for c in answer])


def fixture(tmp: Path, *, spacing=1, engine=None, thresholds=AdvisoryThresholds()):
    records = [
        {"data_id": "a", "question": "q", "response": {"input": "PROMPT:", "content": "abc</think>yes"}, "output_token_ids": [ord(c) for c in "abc</think>yes"], "reasoning_token_count": 3},
        {"data_id": "b", "question": "q", "response": {"input": "PROMPT:", "content": "xyz</think>yes"}, "output_token_ids": [ord(c) for c in "xyz</think>yes"], "reasoning_token_count": 3},
    ]
    return RandomExitReplayEvaluation(records, renderer=Renderer(), artifact_dir=tmp, sampling=SamplingSettings(spacing=spacing, exit_repetitions=2, trial_repetitions=1, answer_budget=10, trial_budget=5), decoder={"temperature": 0}, engine=engine, grade=lambda record, answer: answer == "yes", thresholds=thresholds)


class RandomExitEvaluationTests(unittest.TestCase):
    def test_plan_is_deterministic_and_warnings_do_not_thin_or_block(self):
        with TemporaryDirectory() as directory:
            engine = Engine()
            evaluation = fixture(Path(directory), engine=engine, thresholds=AdvisoryThresholds(generation_calls=1, storage_bytes=1))
            first = evaluation.plan()
            self.assertEqual(first, evaluation.plan())
            self.assertEqual(first["traces"], 2)
            self.assertEqual(first["sampled_positions"], 6)
            self.assertEqual(first["exit_calls"], 12)
            self.assertEqual(first["trial_calls"], 6)
            self.assertEqual(first["maximum_generated_tokens"], 150)
            self.assertTrue(first["warnings"])
            self.assertEqual(asyncio.run(evaluation.collect_missing())["collected"], 18)
            self.assertEqual(asyncio.run(evaluation.collect_missing())["collected"], 0)
            self.assertEqual(len(engine.calls), 18)
            self.assertTrue(all("PROMPT:" in evaluation.renderer.tokenizer.decode(call.prompt_token_ids) for call in engine.calls))

    def test_causal_replay_and_exact_support_without_generation(self):
        with TemporaryDirectory() as directory:
            engine = Engine()
            evaluation = fixture(Path(directory), engine=engine)
            asyncio.run(evaluation.collect_missing())
            count = len(engine.calls)
            method = EarlyExitMethod("stop_two", EveryNTokens(2), lambda ctx: 1.0, lambda score, ctx: score > 0)
            decisions = evaluation.replay([method])
            self.assertEqual([row["position"] for row in decisions], [2, 2])
            self.assertEqual(len(engine.calls), count)
            estimates = evaluation.estimate(decisions)
            self.assertTrue(all(row["exact"]["supported"] for row in estimates))
            self.assertTrue(all(row["exact"]["samples"] == 2 for row in estimates))
            self.assertEqual(evaluation.report(estimates)["methods"]["stop_two"]["estimated_accuracy"], 1)

    def test_trial_is_saved_separately_and_unsampled_stop_is_unsupported(self):
        with TemporaryDirectory() as directory:
            evaluation = fixture(Path(directory), spacing=3, engine=Engine())
            asyncio.run(evaluation.collect_missing())
            method = EarlyExitMethod("trial", AlwaysCheckpoint(), lambda ctx: ctx.agent.generate_temp_answer(ctx.prefix, ctx.position, ctx.kwargs).content, lambda answer, ctx: answer == "{yes}")
            decisions = evaluation.replay([method])
            self.assertEqual(len(decisions), 2)
            self.assertTrue(all(not row.get("supported", True) for row in decisions))
            self.assertTrue(all(not row["exact"]["supported"] for row in evaluation.estimate(decisions)))
            unsupported = SampledExitEstimator().estimate(2, [{"position": 1, "slot": 0, "correct": True, "answer_tokens": 1, "prefill_tokens": 1}])
            self.assertFalse(unsupported["exact"]["supported"])
            self.assertEqual(unsupported["exact"]["reason"], "unsampled_position_or_slot")

    def test_wilson_uncertainty_and_specific_saved_slot(self):
        rows = [
            {"position": 4, "slot": 0, "correct": True, "answer_tokens": 3, "prefill_tokens": 12},
            {"position": 4, "slot": 1, "correct": False, "answer_tokens": 5, "prefill_tokens": 12},
        ]
        estimator = SampledExitEstimator()
        both = estimator.estimate(4, rows)["exact"]
        self.assertEqual(both["estimated_correctness"], 0.5)
        self.assertLess(both["confidence_interval_95"][0], 0.5)
        self.assertGreater(both["confidence_interval_95"][1], 0.5)
        self.assertEqual(estimator.estimate(4, rows, selected_slot=1)["exact"]["estimated_correctness"], 0)

    def test_artifact_identity_feature_versions_and_paired_validation(self):
        with TemporaryDirectory() as directory:
            evaluation = fixture(Path(directory), engine=Engine())
            asyncio.run(evaluation.collect_missing())
            first = evaluation.prepare_features("length-v1", lambda row: len(row["output_token_ids"]))
            self.assertEqual(first, evaluation.prepare_features("length-v1", lambda row: 999))
            self.assertEqual(len(first), 18)
            decisions = evaluation.replay([EarlyExitMethod("early", AlwaysCheckpoint(), lambda ctx: 1, None)])
            estimates = evaluation.estimate(decisions)
            live = [{"data_id": row["data_id"], "method": "early", "correct": True} for row in decisions]
            self.assertEqual(evaluation.report(estimates, paired_live_rows=live)["paired_live_validation"]["mean_absolute_error"], 0)
            path = next((Path(directory) / "generations").rglob("*.json"))
            row = json.loads(path.read_text())
            row["identity"] = "wrong"
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Artifact identity mismatch"):
                evaluation.prepare_features("length-v2", lambda item: len(item["output_token_ids"]))


if __name__ == "__main__":
    unittest.main()
