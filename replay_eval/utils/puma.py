"""PUMA verifier helpers."""

from __future__ import annotations

import difflib
import logging
import threading
import time
from typing import Any


DEFAULT_PUMA_EMBEDDING_MODEL = "ZhishanQ/qwen3-embedding-redundancy-detector-0.6B"
DEFAULT_PUMA_EMBEDDING_MAX_TOKENS = 2048
logger = logging.getLogger(__name__)
_PUMA_SIMILARITY_BACKENDS: dict[
    tuple[str, str | None, int, int | None], "PUMASimilarityBackend"
] = {}
_PUMA_SIMILARITY_BACKENDS_LOCK = threading.Lock()

CONFIDENT_ENDINGS: dict[str, str] = {
    "math": "\n**Final Answer**\n\nThe final answer is \\boxed",
    "gpqa": "\n**Final Answer**\n\nThe answer choice is \\boxed",
    "code": "</think>\n\n### Solution Code\n```python\n",
}


class PUMASimilarityBackend:
    def __init__(
        self,
        model_name: str = DEFAULT_PUMA_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_tokens: int | None = DEFAULT_PUMA_EMBEDDING_MAX_TOKENS,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError from e
        self.device = resolve_embedding_device(device)
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        model_source = resolve_cached_embedding_model(model_name)
        start_time = time.perf_counter()
        logger.info(
            "Loading PUMA embedding model=%s source=%s device=%s max_tokens=%s",
            model_name,
            model_source,
            self.device or "auto",
            self.max_tokens,
        )
        self._model = SentenceTransformer(model_source, device=self.device)
        if self.max_tokens is not None:
            self._model.max_seq_length = self.max_tokens
        logger.info(
            "Loaded PUMA embedding model=%s elapsed=%.2fs",
            model_name,
            time.perf_counter() - start_time,
        )
        self._cache: dict[str, Any] = {}
        # The backend is process-global and therefore shared by agents running
        # in different runner threads. SentenceTransformer's fast tokenizer is
        # not safe for concurrent encode calls.
        self._operation_lock = threading.RLock()

    def _truncate_text_for_embedding(self, text: str) -> str:
        if self.max_tokens is None or not text:
            return text

        tokenizer = getattr(self._model, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        decode = getattr(tokenizer, "decode", None)
        if callable(encode) and callable(decode):
            try:
                token_ids = encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_tokens,
                )
                if len(token_ids) <= self.max_tokens:
                    return decode(token_ids, skip_special_tokens=True)
            except TypeError:
                try:
                    token_ids = encode(text, add_special_tokens=False)[: self.max_tokens]
                    return decode(token_ids, skip_special_tokens=True)
                except Exception:
                    pass
            except Exception:
                pass

        max_chars = self.max_tokens * 4
        return text[:max_chars]

    def _prepare_embedding_texts(self, texts: list[str]) -> list[str]:
        return [self._truncate_text_for_embedding(text) for text in texts]

    def _encode(self, text: str):
        with self._operation_lock:
            if text not in self._cache:
                self._cache[text] = self._model.encode(
                    self._truncate_text_for_embedding(text),
                    normalize_embeddings=True,
                    convert_to_tensor=True,
                    device=self.device,
                )
            return self._cache[text]

    def max_similarity(self, current: str, previous: list[str]) -> float:
        with self._operation_lock:
            if not current or not previous:
                return 0.0
            cur_emb = self._encode(current)
            best = 0.0
            for prev in previous:
                if not prev:
                    continue
                prev_emb = self._encode(prev)
                sim = _score_to_cpu_float(cur_emb @ prev_emb)
                if sim > best:
                    best = sim
            return best

    def max_similarities(self, steps: list[str], window_size: int) -> list[float | None]:
        """Compute PUMA max previous-step similarities for one full trajectory."""
        with self._operation_lock:
            return self._max_similarities_locked(steps, window_size)

    def _max_similarities_locked(
        self, steps: list[str], window_size: int
    ) -> list[float | None]:
        if not steps:
            return []

        start_time = time.perf_counter()
        logger.info(
            "PUMA embedding encode started steps=%s batch_size=%s device=%s max_tokens=%s",
            len(steps),
            self.batch_size,
            self.device or "auto",
            self.max_tokens,
        )
        embedding_steps = self._prepare_embedding_texts(steps)
        embeddings = self._model.encode(
            embedding_steps,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        logger.info(
            "PUMA embedding encode finished steps=%s elapsed=%.2fs",
            len(steps),
            time.perf_counter() - start_time,
        )
        for step, embedding in zip(steps, embeddings):
            self._cache.setdefault(step, embedding)

        max_sims: list[float | None] = [None]
        for i in range(1, len(steps)):
            start = 0 if window_size <= 0 else max(0, i - window_size)
            best = None
            for j in range(start, i):
                sim = _score_to_cpu_float(embeddings[i] @ embeddings[j])
                best = sim if best is None or sim > best else best
            max_sims.append(best)
        return max_sims

    def reset(self) -> None:
        with self._operation_lock:
            self._cache.clear()


def get_puma_similarity_backend(
    model_name: str = DEFAULT_PUMA_EMBEDDING_MODEL,
    *,
    device: str | None = None,
    batch_size: int = 32,
    max_tokens: int | None = DEFAULT_PUMA_EMBEDDING_MAX_TOKENS,
) -> PUMASimilarityBackend:
    """Return a process-local shared PUMA backend.

    Runner creates a fresh agent per record, so without this cache each PUMA
    verifier reloads the embedding model for every question.
    """
    resolved_device = resolve_embedding_device(device)
    resolved_max_tokens = int(max_tokens) if max_tokens is not None else None
    key = (model_name, resolved_device, int(batch_size), resolved_max_tokens)
    with _PUMA_SIMILARITY_BACKENDS_LOCK:
        backend = _PUMA_SIMILARITY_BACKENDS.get(key)
        if backend is None:
            backend = PUMASimilarityBackend(
                model_name,
                device=resolved_device,
                batch_size=int(batch_size),
                max_tokens=resolved_max_tokens,
            )
            _PUMA_SIMILARITY_BACKENDS[key] = backend
        return backend


def resolve_embedding_device(device: str | None) -> str | None:
    if not device:
        return None
    normalized = device.lower()
    if normalized != "auto":
        return normalized

    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_cached_embedding_model(model_name: str) -> str:
    """Prefer an existing local Hugging Face snapshot over remote metadata checks."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name

    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception as exc:
        logger.info(
            "No local Hugging Face snapshot for PUMA embedding model=%s; using model id: %s",
            model_name,
            exc,
        )
        return model_name


def get_confident_ending(task_type: str) -> str:
    return CONFIDENT_ENDINGS.get(task_type, CONFIDENT_ENDINGS["math"])


def extract_boxed_answer(text: str) -> str:
    idx = text.find("\\boxed{")
    if idx == -1:
        idx = text.find("boxed{")
        if idx == -1:
            return ""
        start = idx + len("boxed{")
    else:
        start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return text[start:].strip()


def extract_trial_code(text: str) -> str:
    if "```" in text:
        return text[: text.index("```")].strip()
    return text.strip()


def extract_trial_answer(text: str, task_type: str) -> str:
    if task_type == "code":
        return extract_trial_code(text)
    return extract_boxed_answer(text)


def answers_match(
    a: Any,
    b: Any,
    task_type: str,
    code_threshold: float = 0.8,
) -> bool:
    if not a or not b:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return a is True and b is True
    if task_type == "code":
        return difflib.SequenceMatcher(None, a, b).ratio() >= code_threshold
    return a == b


def best_trial_confidence(trial_results: list[dict]) -> float:
    eligible_trials = [
        trial
        for trial in trial_results
        if trial.get("collected_answer", trial.get("answer"))
    ]
    if not eligible_trials:
        return 0.0
    return max(float(trial["confidence"]) for trial in eligible_trials)


def loop_breaker_fires(
    *,
    step_index: int,
    consecutive_redundant: int,
    trial_results: list[dict],
    consecutive_redundancy_stop: int,
    min_loop_breaker_step: int,
    forced_stop_min_confidence: float,
) -> bool:
    if consecutive_redundancy_stop <= 0:
        return False
    if step_index < min_loop_breaker_step:
        return False
    if consecutive_redundant < consecutive_redundancy_stop:
        return False
    return best_trial_confidence(trial_results) >= forced_stop_min_confidence


def should_exit_verified(
    trial_results: list[dict],
    *,
    task_type: str,
    consecutive: int,
    confidence_threshold: float,
    epsilon: float,
    code_similarity_threshold: float,
) -> bool:
    n = len(trial_results)
    if n < consecutive:
        return False

    for i in range(n - consecutive + 1):
        first = trial_results[i]
        if first["confidence"] < confidence_threshold:
            continue
        first_answer = first.get("collected_answer", first["answer"])
        first_conf = first["confidence"]
        if not first_answer:
            continue

        ok = True
        for j in range(1, consecutive):
            entry = trial_results[i + j]
            if entry["step"] != first["step"] + j:
                ok = False
                break
            if not answers_match(
                entry.get("collected_answer", entry["answer"]),
                first_answer,
                task_type,
                code_similarity_threshold,
            ):
                ok = False
                break
            if entry["confidence"] < first_conf - epsilon:
                ok = False
                break

        if ok:
            return True
    return False


def _score_to_cpu_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
