"""On-policy distillation (OPD) helpers for async GRPO.

Teacher routing, config helpers, and remote teacher serve resolution.
Advantage computation lives in advantage_estimator.OPDAdvantageEstimator.
IS truncation lives in loss_functions.ClippedPGLoss (ICE-POP mode).

The OPD teacher is served by an externally-managed vLLM endpoint (each
``_teachers.<alias>`` entry carries a ``url`` and a ``model``). This module
scores trajectories by POSTing the student-generated token sequences to the
OpenAI-compatible ``/v1/completions`` endpoint with ``echo=true`` and
``max_tokens=0`` (score-only prefill, no decode). The response carries the
logprob of the actual next token at every position, which matches the
convention used by the rest of the training loop: the logprob of input token
``i`` is stored at position ``i``, and position 0 is 0.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, NotRequired, Optional, TypedDict

import aiohttp
import torch


# ---------------------------------------------------------------------------
# Config TypedDicts
# ---------------------------------------------------------------------------


class OrmEstimatorConfig(TypedDict):
    orm_estimator_name: str
    use_leave_one_out_baseline: NotRequired[bool]
    normalize_rewards: NotRequired[bool]
    minus_baseline: NotRequired[bool]


class OnPolicyDistillationConfig(TypedDict):
    enabled: bool
    # Values are `_teachers.<alias>` entries: {url, model} serve specs.
    teacher_model_by_agent_name: NotRequired[dict[str, Any]]
    default_teacher_alias: NotRequired[Optional[str]]
    strict_agent_name_match: NotRequired[bool]
    opd_advantage_weight: NotRequired[float]
    orm_advantage_weight: NotRequired[float]
    opd_advantage_clip_low: NotRequired[float]
    opd_advantage_clip_high: NotRequired[float]
    orm_advantage_clip_low: NotRequired[float]
    orm_advantage_clip_high: NotRequired[float]
    zero_out_of_bounds_advantages: NotRequired[bool]
    orm_advantage_estimator: NotRequired[OrmEstimatorConfig]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def is_opd_enabled(master_config: dict[str, Any]) -> bool:
    return bool(master_config.get("on_policy_distillation", {}).get("enabled", False))


# ---------------------------------------------------------------------------
# Teacher serve specs
# ---------------------------------------------------------------------------


def normalize_teacher_serve(raw: Any) -> dict[str, str]:
    """Validate a ``_teachers`` entry and coerce it into ``{url, model}``.

    The value may be an OmegaConf ``DictConfig`` rather than a plain dict when
    it comes straight from the YAML config, so no ``isinstance`` checks are
    made here — bracket access and string conversion handle both.
    """
    try:
        url = raw["url"]
        model = raw["model"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "OPD teacher entries must be mappings with 'url' and 'model' "
            f"fields, got {raw!r}"
        ) from exc
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"OPD teacher 'url' must be a non-empty string, got {url!r}")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"OPD teacher 'model' must be a non-empty string, got {model!r}")
    return {"url": url.strip(), "model": model.strip()}


def probe_teacher_serve(url: str, models: set[str], timeout_s: float = 30.0) -> None:
    """GET ``{url}/v1/models`` and verify every expected ``model`` is served.

    Called at setup time, before any rollout is collected, so a bad URL or
    model name fails loudly instead of degrading every trajectory group at
    collection time.
    """
    models_url = f"{url.rstrip('/')}/v1/models"
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(
            models_url, timeout=timeout_s, context=ssl_ctx
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(
            f"OPD teacher serve health check failed for '{url}': {exc}"
        ) from exc
    served = {m.get("id") for m in payload.get("data", []) if isinstance(m, dict)}
    missing = models - served
    if missing:
        raise RuntimeError(
            f"OPD teacher serve at '{url}' does not serve model(s) "
            f"{sorted(missing)}. Served models: {sorted(served)}"
        )


# ---------------------------------------------------------------------------
# Teacher routing
# ---------------------------------------------------------------------------


def resolve_teacher_specs(
    agent_refs: list[dict],
    teacher_model_by_agent_name: dict[str, Any],
    default_teacher_alias: Optional[str] = None,
    strict_agent_name_match: bool = False,
) -> list[dict[str, str]]:
    """Resolve agent references to normalized ``{url, model}`` serve specs.

    Each agent reference is mapped to a teacher alias (its own ``name``, or
    ``default_teacher_alias`` when unmapped), then to the normalized serve
    spec stored under that alias.
    """
    specs: list[dict[str, str]] = []
    for ref in agent_refs:
        name = ref["name"]
        if name in teacher_model_by_agent_name:
            alias = name
        elif strict_agent_name_match:
            raise ValueError(
                f"No teacher model mapping for agent '{name}'. "
                f"Available: {sorted(teacher_model_by_agent_name.keys())}"
            )
        elif default_teacher_alias:
            print(f"[OPD] Agent '{name}' not in teacher mapping, falling back to '{default_teacher_alias}'")
            alias = default_teacher_alias
        else:
            raise ValueError(
                f"No teacher model mapping for agent '{name}' and no default_teacher_alias set."
            )
        if alias not in teacher_model_by_agent_name:
            raise KeyError(
                f"Agent alias '{alias}' has no teacher serve mapping. "
                f"Available: {sorted(teacher_model_by_agent_name.keys())}"
            )
        specs.append(normalize_teacher_serve(teacher_model_by_agent_name[alias]))
    return specs


def validate_teacher_serves(master_config: dict[str, Any]) -> None:
    """Fail fast if any OPD teacher mapping is not a ``{url, model}`` serve.

    Also probes each unique serve URL (``GET {url}/v1/models``) so a bad URL or
    model name fails at setup, before any rollout is collected.
    """
    opd_cfg = master_config["on_policy_distillation"]
    teacher_model_by_agent_name = opd_cfg["teacher_model_by_agent_name"]
    models_by_url: dict[str, set[str]] = {}
    for raw in teacher_model_by_agent_name.values():
        spec = normalize_teacher_serve(raw)
        models_by_url.setdefault(spec["url"], set()).add(spec["model"])
    for url, models in models_by_url.items():
        probe_teacher_serve(url, models)


# ---------------------------------------------------------------------------
# Remote vLLM teacher client
# ---------------------------------------------------------------------------

# vLLM treats the literal "EMPTY" api key as no authentication.
_API_KEY = "EMPTY"
# Soft cap on concurrent in-flight scoring requests (connector limit).
_MAX_CONCURRENT_REQUESTS = 64
# Total per-request timeout; prompts up to max_total_sequence_length tokens
# can keep the serve busy for a long time. A timeout is a signal to retry.
_REQUEST_TIMEOUT_S = 1800
# HTTP statuses that mean "try again later": the serve is reachable but
# temporarily unable to score this request. Retried without bound.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRY_BACKOFF_BASE_S = 2.0
# Cap on the exponential backoff: delays go 2, 4, 8, 16, 32, 60, 60, ...
_MAX_RETRY_BACKOFF_S = 60.0
# Substrings in a 4xx body that identify a permanent context-length overflow
# (retrying can never succeed).
_CONTEXT_LENGTH_SUBSTRINGS = (
    "Please reduce the length of the input prompt or the number of requested output tokens",
    "maximum context length",
    "the input is too long",
)


class TeacherContextLengthError(RuntimeError):
    """Permanent: the trajectory group does not fit the teacher context.

    Raised on first detection and never retried — a retry can never succeed.
    """

    def __init__(self, url: str, body: str):
        super().__init__(
            f"Teacher serve {url} cannot fit the trajectory group "
            f"(context-length overflow): {body[:500]}"
        )
        self.url = url
        self.body = body


class _TeacherServerError(Exception):
    """Raised when the teacher serve replies with an HTTP error status."""

    def __init__(self, status: int, body: str):
        super().__init__(f"Teacher serve returned HTTP {status}: {body[:1000]}")
        self.status = status
        self.body = body


class _TeacherWorkerDied(RuntimeError):
    """Set on in-flight futures when the worker thread dies."""


def _retry_delay(attempt: int) -> float:
    """Exponential backoff delay for retry ``attempt``, capped at 60s.

    ±20% uniform jitter desynchronizes the concurrently retrying prompt
    groups so they don't all re-fire on the same 60s tick when the serve
    recovers (lockstep retries would thundering-herd it straight back into
    failure).
    """
    base = min(_RETRY_BACKOFF_BASE_S * (2 ** attempt), _MAX_RETRY_BACKOFF_S)
    return base * random.uniform(0.8, 1.2)


class VLLMTeacherLogprobClient:
    """HTTP client that scores input token sequences on remote teacher serves.

    A single long-lived worker thread owns one asyncio event loop and one
    aiohttp session for the whole client lifetime. ``score_group`` may be
    called concurrently from any thread/loop: it hands the scoring coroutine
    to the worker loop and awaits the result. This gives real connection
    reuse across all prompt groups and removes the per-loop session cache
    hazard (a cached session bound to a dead loop raises ``RuntimeError`` on
    every subsequent retry).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ready = threading.Event()
        self._pending_futures: set = set()
        self._closed = False

    def _ensure_worker(self) -> asyncio.AbstractEventLoop:
        """Return the worker loop, spawning the worker thread if needed."""
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("VLLMTeacherLogprobClient is closed")
                thread = self._thread
                loop = self._loop
                if (
                    thread is not None
                    and thread.is_alive()
                    and loop is not None
                    and not loop.is_closed()
                ):
                    return loop
                if thread is None or not thread.is_alive():
                    # No healthy worker: start a new one. A live thread with a
                    # missing loop is mid-startup — wait for it instead of
                    # spawning a duplicate (two workers would clobber the
                    # shared session and serve cross-loop requests).
                    self._ready.clear()
                    self._thread = threading.Thread(
                        target=self._worker_main,
                        daemon=True,
                        name="opd-teacher-client",
                    )
                    self._thread.start()
            while self._thread.is_alive() and not self._ready.is_set():
                self._ready.wait(0.1)
            loop = self._loop
            if loop is not None and not loop.is_closed():
                return loop
            # Worker died before becoming ready; respawn.
        # The loop above respawns forever; this line is defensive only.
        raise RuntimeError("VLLMTeacherLogprobClient failed to start its worker thread")

    def _worker_main(self) -> None:
        loop = asyncio.new_event_loop()
        session: aiohttp.ClientSession | None = None
        try:
            session_holder: dict[str, aiohttp.ClientSession] = {}

            async def _create_session() -> None:
                connector = aiohttp.TCPConnector(
                    limit=_MAX_CONCURRENT_REQUESTS, keepalive_timeout=676767.0, ssl=False
                )
                session_holder["session"] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S),
                )

            loop.run_until_complete(_create_session())
            session = session_holder["session"]
        except Exception as exc:
            # Log instead of dying silently: a failed session creation would
            # otherwise look like a hang, with _ensure_worker respawning the
            # thread (and failing again) on every score_group call.
            print(
                f"[OPD teacher] failed to create aiohttp session; the worker "
                f"thread will be restarted on the next score_group call: {exc}",
                flush=True,
            )
            loop.close()
            return

        self._session = session
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            # Best-effort teardown; in-flight callers must fail fast instead
            # of hanging on a dead loop.
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                if self._loop is loop:
                    self._loop = None
                    self._session = None
                    self._ready.clear()
                stale = [
                    entry for entry in self._pending_futures if entry[1] is loop
                ]
                for entry in stale:
                    self._pending_futures.discard(entry)
                for fut, _ in stale:
                    if not fut.done():
                        fut.set_exception(
                            _TeacherWorkerDied(
                                "vLLM teacher worker thread died; the next "
                                "score_group call will restart it"
                            )
                        )
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            if tasks:
                try:
                    loop.run_until_complete(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )
                except Exception:
                    pass
            try:
                if session is not None and not session.closed:
                    loop.run_until_complete(session.close())
            except Exception:
                pass
            loop.close()

    async def score_group(
        self,
        teacher: dict[str, str],
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score a batch of trajectories on the given teacher serve.

        Args:
            teacher: Normalized serve spec with ``url`` and ``model``.
            input_ids: [B, S] right-padded input token ids.
            input_lengths: [B] per-row token lengths. Rows are trimmed to
                their length so the serve does not prefill padding.

        Returns:
            [B, S] float32 tensor with the teacher logprob of the input token
            at every position (position 0 is 0), right-padded to S.
        """
        loop = self._ensure_worker()
        raw = asyncio.run_coroutine_threadsafe(
            self._score_on_loop(teacher, input_ids, input_lengths), loop
        )
        # The raw future is owned by the worker loop's teardown; give callers
        # their own future so a dying worker can set _TeacherWorkerDied on it
        # without racing the raw future's chained callbacks.
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def _propagate(src: concurrent.futures.Future) -> None:
            with self._lock:
                if fut.done():
                    return
                try:
                    exc = src.exception()
                except concurrent.futures.CancelledError:
                    fut.cancel()
                    return
                if exc is not None:
                    fut.set_exception(exc)
                else:
                    fut.set_result(src.result())

        raw.add_done_callback(_propagate)
        entry = (fut, loop)
        with self._lock:
            self._pending_futures.add(entry)
            if self._loop is not loop or not self._thread.is_alive():
                # The worker died between _ensure_worker and scheduling; the
                # coroutine can never run, so fail the future ourselves.
                if not fut.done():
                    fut.set_exception(
                        _TeacherWorkerDied(
                            "vLLM teacher worker thread died before the "
                            "score_group request could be served"
                        )
                    )
        try:
            return await asyncio.wrap_future(fut)
        finally:
            with self._lock:
                self._pending_futures.discard(entry)

    async def _score_on_loop(
        self,
        teacher: dict[str, str],
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Worker-loop side of :meth:`score_group` (touches the session only here)."""
        batch_size, padded_len = input_ids.shape
        lengths = (
            input_lengths.tolist()
            if input_lengths is not None
            else [padded_len] * batch_size
        )
        prompts = [input_ids[i, : lengths[i]].tolist() for i in range(batch_size)]

        payload = {
            "model": teacher["model"],
            "prompt": prompts,
            "echo": True,
            "max_tokens": 0,
            "logprobs": 1,
            "return_token_ids": True,
            "temperature": 0,
            "add_special_tokens": False,
        }
        headers: dict[str, str] = {"Authorization": f"Bearer {_API_KEY}"}

        session = self._session
        if session is None:
            raise _TeacherWorkerDied(
                "vLLM teacher worker session is gone; the worker thread died"
            )
        response_json = await self._post_with_retries(
            session, teacher, payload, headers
        )

        choices = response_json["choices"]
        if not isinstance(choices, list) or len(choices) != batch_size:
            got = len(choices) if isinstance(choices, list) else "non-list"
            raise RuntimeError(
                f"Teacher serve returned {got} choices for {batch_size} prompts"
            )

        out = torch.zeros(batch_size, padded_len, dtype=torch.float32)
        for i, choice in enumerate(choices):
            row = self._parse_choice(choice, prompts[i], teacher)
            out[i, : row.numel()] = row
        return out

    @staticmethod
    def _parse_choice(
        choice: dict[str, Any],
        sent_ids: list[int],
        teacher: dict[str, str],
    ) -> torch.Tensor:
        """Extract the per-position logprobs for one scored prompt."""
        echo_ids = choice.get("prompt_token_ids")
        if echo_ids is not None and echo_ids != sent_ids:
            raise RuntimeError(
                f"Teacher serve {teacher['url']} echoed mismatched prompt token "
                f"ids (sent {len(sent_ids)}, got {len(echo_ids)})"
            )
        logprobs = choice.get("logprobs")
        if logprobs is None:
            raise RuntimeError(
                f"Teacher serve {teacher['url']} returned no logprobs "
                "(is prompt logprob scoring enabled?)"
            )
        token_logprobs = logprobs.get("token_logprobs")
        if token_logprobs is None or len(token_logprobs) != len(sent_ids):
            got = len(token_logprobs) if token_logprobs is not None else 0
            raise RuntimeError(
                f"Teacher serve {teacher['url']} returned {got} token logprobs "
                f"for a {len(sent_ids)}-token prompt"
            )
        # Position i holds the logprob of input token i (position 0 is None
        # because the first token has no prefix, so it becomes 0).
        row = []
        for i, v in enumerate(token_logprobs):
            if v is None and i == 0:
                row.append(0.0)
            elif v is None:
                # A None at a non-zero position is unexpected — the serve did
                # not produce a logprob for this token. Mask it with NaN
                # (zeroed downstream by OPDAdvantageEstimator) instead of
                # silently treating it as logprob 0, which would fabricate a
                # huge positive OPD advantage at this position.
                print(
                    f"[OPD teacher] {teacher['url']} returned None logprob at "
                    f"position {i} of a {len(token_logprobs)}-token prompt; "
                    f"masking this position with NaN",
                    flush=True,
                )
                row.append(float("nan"))
            else:
                row.append(float(v))
        return torch.tensor(row, dtype=torch.float32)

    async def _post_with_retries(
        self,
        session: aiohttp.ClientSession,
        teacher: dict[str, str],
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """POST one scoring request, retrying transient failures without bound.

        Transient failures (timeouts, transport errors, JSON decode errors,
        429/5xx) are retried forever with exponential backoff capped at 60s —
        the teacher scoring is a delivery guarantee. Permanent failures (other
        4xx and context-length overflow) raise immediately.
        """
        completions_url = f"{teacher['url'].rstrip('/')}/v1/completions"
        attempt = 0
        t_start = time.monotonic()
        # Retrying forever is intentional: teacher scoring is a delivery
        # guarantee. A down teacher serve must stall the pipeline — the prompt
        # group stays in the collector while we retry — rather than let
        # training proceed without the teacher signal. Stalling is visible
        # (buffer starvation) and safe (nothing is dropped or corrupted);
        # skipping or dropping the group would silently degrade the OPD
        # objective.
        while True:
            err: Exception | None = None
            try:
                async with session.post(
                    completions_url, json=payload, headers=headers
                ) as resp:
                    body = await resp.text()
                    if resp.status in _RETRYABLE_STATUS_CODES:
                        raise _TeacherServerError(resp.status, body)  # transient
                    if resp.status >= 400:
                        for sub in _CONTEXT_LENGTH_SUBSTRINGS:
                            if sub in body:
                                raise TeacherContextLengthError(
                                    teacher["url"], body
                                )  # permanent
                        raise _TeacherServerError(resp.status, body)  # permanent
                    return json.loads(body)
            except TeacherContextLengthError:
                raise
            except _TeacherServerError as exc:
                if exc.status not in _RETRYABLE_STATUS_CODES:
                    raise
                err = exc
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
                err = exc
            elapsed = time.monotonic() - t_start
            print(
                f"[OPD teacher] {teacher['url']} attempt {attempt} failed "
                f"({elapsed:.0f}s so far): {err} — retrying"
            )
            await asyncio.sleep(_retry_delay(attempt))
            attempt += 1
        # The loop above retries forever; this line is defensive only.
        raise RuntimeError("unreachable: teacher scoring retries without bound")

    async def close(self, timeout: float = 5.0) -> None:
        """Stop the worker thread and close its loop/session.

        Best-effort: joins the worker with ``timeout`` seconds. Used by tests;
        the collector keeps the worker alive for its own lifetime.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            loop = self._loop
        if thread is not None and loop is not None and thread.is_alive():
            def _stop() -> None:
                loop.stop()

            try:
                loop.call_soon_threadsafe(_stop)
            except RuntimeError:
                pass
            thread.join(timeout)
