# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import threading
import urllib.error

import pytest
import torch

from nemo_rl.algorithms import vllm_teacher_client as client_module
from nemo_rl.algorithms.vllm_teacher_client import (
    TeacherContextLengthError,
    VLLMTeacherLogprobClient,
    normalize_teacher_serve,
    probe_teacher_serve,
)

_TEACHER = {"url": "http://teacher:8000/v1", "model": "teacher-model"}


def _run(coro):
    return asyncio.run(coro)


def _run_with_close(client, coro):
    """Run a coroutine and close the client worker afterwards."""

    async def _inner():
        try:
            return await coro
        finally:
            await client.close()

    return asyncio.run(_inner())


def _canned_response(prompts, fill=0.5):
    choices = []
    for p in prompts:
        token_logprobs = [None] + [fill] * (len(p) - 1)
        choices.append(
            {"prompt_token_ids": p, "logprobs": {"token_logprobs": token_logprobs}}
        )
    return {"choices": choices}


class _FakePoster:
    """Records payloads and returns canned JSON responses in order."""

    def __init__(self, *payloads):
        self._payloads = list(payloads)
        self.calls = []

    async def __call__(self, session, teacher, payload, headers):
        self.calls.append((teacher, payload, headers))
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _client_with(poster):
    client = VLLMTeacherLogprobClient()
    client._post_with_retries = poster
    return client


class _FakeResponse:
    def __init__(self, status: int, body: str):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def text(self) -> str:
        return self._body


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in for _post_with_retries tests."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def post(self, url, json=None, headers=None):
        self.requests.append((url, json, headers))
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# normalize_teacher_serve
# ---------------------------------------------------------------------------


def test_normalize_teacher_serve_valid():
    assert normalize_teacher_serve({"url": "http://x/v1", "model": "m"}) == {
        "url": "http://x/v1",
        "model": "m",
    }


def test_normalize_teacher_serve_strips_whitespace():
    assert normalize_teacher_serve({"url": " http://x/v1 ", "model": " m "}) == {
        "url": "http://x/v1",
        "model": "m",
    }


def test_normalize_teacher_serve_missing_fields():
    with pytest.raises(ValueError, match="'url' and 'model'"):
        normalize_teacher_serve({"url": "http://x/v1"})


def test_normalize_teacher_serve_not_a_mapping():
    with pytest.raises(ValueError, match="mappings with 'url' and 'model'"):
        normalize_teacher_serve("/ckpt/path")


def test_normalize_teacher_serve_non_string_fields():
    with pytest.raises(ValueError, match="'url'"):
        normalize_teacher_serve({"url": None, "model": "m"})
    with pytest.raises(ValueError, match="'model'"):
        normalize_teacher_serve({"url": "http://x/v1", "model": 42})


def test_normalize_teacher_serve_empty_strings():
    with pytest.raises(ValueError, match="'url'"):
        normalize_teacher_serve({"url": "   ", "model": "m"})
    with pytest.raises(ValueError, match="'model'"):
        normalize_teacher_serve({"url": "http://x/v1", "model": ""})


# ---------------------------------------------------------------------------
# probe_teacher_serve
# ---------------------------------------------------------------------------


class _FakeURLResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self._payload


def test_probe_teacher_serve_success(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append((url, timeout))
        return _FakeURLResponse({"data": [{"id": "teacher-model"}]})

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    probe_teacher_serve("http://teacher:8000/v1", {"teacher-model"})

    assert calls == [("http://teacher:8000/v1/models", 30.0)]


def test_probe_teacher_serve_missing_model(monkeypatch):
    def fake_urlopen(url, timeout=None):
        return _FakeURLResponse({"data": [{"id": "other-model"}]})

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="does not serve"):
        probe_teacher_serve("http://teacher:8000/v1", {"teacher-model"})


def test_probe_teacher_serve_connection_error(monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(client_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="health check failed"):
        probe_teacher_serve("http://teacher:8000/v1", {"teacher-model"})


# ---------------------------------------------------------------------------
# _retry_delay — jitter
# ---------------------------------------------------------------------------


def test_retry_delay_jitter_bounds(monkeypatch):
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 2.0)
    for attempt in range(6):
        base = min(2.0 * (2 ** attempt), client_module._MAX_RETRY_BACKOFF_S)
        for _ in range(50):
            delay = client_module._retry_delay(attempt)
            assert base * 0.8 <= delay <= base * 1.2


# ---------------------------------------------------------------------------
# score_group — payload construction
# ---------------------------------------------------------------------------


def test_score_group_payload_and_parse():
    """Payload uses echo + max_tokens=0 and rows are trimmed to their lengths."""
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    lengths = torch.tensor([3, 5])
    poster = _FakePoster(_canned_response([[1, 2, 3], [4, 5, 6, 7, 8]], fill=0.5))
    client = _client_with(poster)

    out = _run_with_close(client, client.score_group(_TEACHER, input_ids, lengths))

    assert len(poster.calls) == 1
    teacher, payload, headers = poster.calls[0]
    assert teacher == _TEACHER
    assert payload["model"] == "teacher-model"
    assert payload["prompt"] == [[1, 2, 3], [4, 5, 6, 7, 8]]
    assert payload["echo"] is True
    assert payload["max_tokens"] == 0
    assert payload["logprobs"] == 1
    assert payload["return_token_ids"] is True
    assert payload["temperature"] == 0
    assert payload["add_special_tokens"] is False
    assert headers == {"Authorization": "Bearer EMPTY"}

    expected = torch.tensor(
        [[0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.5, 0.5]]
    )
    assert torch.allclose(out, expected)


def test_score_group_without_lengths_uses_full_rows():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    poster = _FakePoster(_canned_response([[1, 2, 3], [4, 5, 6]]))
    client = _client_with(poster)

    out = _run_with_close(client, client.score_group(_TEACHER, input_ids))

    _, payload, _ = poster.calls[0]
    assert payload["prompt"] == [[1, 2, 3], [4, 5, 6]]
    expected = torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# score_group — response validation
# ---------------------------------------------------------------------------


def test_score_group_echo_mismatch_raises():
    input_ids = torch.tensor([[1, 2, 3]])
    bad = {"choices": [{"prompt_token_ids": [9, 9, 9], "logprobs": {"token_logprobs": [None, 0.1, 0.1]}}]}
    client = _client_with(_FakePoster(bad))

    with pytest.raises(RuntimeError, match="mismatched prompt token"):
        _run_with_close(client, client.score_group(_TEACHER, input_ids))


def test_score_group_missing_logprobs_raises():
    input_ids = torch.tensor([[1, 2, 3]])
    bad = {"choices": [{"prompt_token_ids": [1, 2, 3], "logprobs": None}]}
    client = _client_with(_FakePoster(bad))

    with pytest.raises(RuntimeError, match="no logprobs"):
        _run_with_close(client, client.score_group(_TEACHER, input_ids))


def test_score_group_wrong_logprob_count_raises():
    input_ids = torch.tensor([[1, 2, 3]])
    bad = {"choices": [{"prompt_token_ids": [1, 2, 3], "logprobs": {"token_logprobs": [None]}}]}
    client = _client_with(_FakePoster(bad))

    with pytest.raises(RuntimeError, match="token logprobs"):
        _run_with_close(client, client.score_group(_TEACHER, input_ids))


def test_score_group_mid_sequence_none_becomes_nan():
    """A None at a non-zero position is masked with NaN, not treated as 0."""
    input_ids = torch.tensor([[1, 2, 3]])
    bad = {
        "choices": [
            {
                "prompt_token_ids": [1, 2, 3],
                "logprobs": {"token_logprobs": [None, None, 0.5]},
            }
        ]
    }
    client = _client_with(_FakePoster(bad))

    out = _run_with_close(client, client.score_group(_TEACHER, input_ids))

    assert out[0, 0] == 0.0
    assert torch.isnan(out[0, 1])
    assert out[0, 2] == 0.5


def test_score_group_choice_count_mismatch_raises():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    client = _client_with(_FakePoster({"choices": []}))

    with pytest.raises(RuntimeError, match="choices"):
        _run_with_close(client, client.score_group(_TEACHER, input_ids))


# ---------------------------------------------------------------------------
# _post_with_retries — retry semantics
# ---------------------------------------------------------------------------


def test_post_with_retries_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    client = VLLMTeacherLogprobClient()
    session = _FakeSession(
        _FakeResponse(503, "unavailable"),
        _FakeResponse(200, '{"choices": []}'),
    )

    result = _run(
        client._post_with_retries(
            session, _TEACHER, {"prompt": [[1]]}, {"Authorization": "Bearer EMPTY"}
        )
    )

    assert result == {"choices": []}
    assert len(session.requests) == 2
    url, payload, headers = session.requests[0]
    assert url == "http://teacher:8000/v1/completions"


def test_retryable_status_retried_forever(monkeypatch):
    """Retryable statuses are retried until success — no attempt cap."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    client = VLLMTeacherLogprobClient()
    session = _FakeSession(
        _FakeResponse(503, "unavailable"),
        _FakeResponse(503, "unavailable"),
        _FakeResponse(200, '{"choices": []}'),
    )

    result = _run(
        client._post_with_retries(session, _TEACHER, {"prompt": [[1]]}, {})
    )

    assert result == {"choices": []}
    assert len(session.requests) == 3


def test_permanent_4xx_raises_immediately(monkeypatch):
    """A 4xx without context-length substrings raises on the first attempt."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    client = VLLMTeacherLogprobClient()
    session = _FakeSession(_FakeResponse(400, "model not found"))

    with pytest.raises(client_module._TeacherServerError, match="HTTP 400"):
        _run(client._post_with_retries(session, _TEACHER, {"prompt": [[1]]}, {}))

    assert len(session.requests) == 1


def test_context_length_raises_no_retry(monkeypatch):
    """A context-length overflow body raises TeacherContextLengthError, no retry."""
    monkeypatch.setattr(client_module, "_RETRY_BACKOFF_BASE_S", 0.0)
    client = VLLMTeacherLogprobClient()
    body = "This model's maximum context length is 4096 tokens and the input is too long."
    session = _FakeSession(_FakeResponse(400, body))

    with pytest.raises(TeacherContextLengthError, match="context-length overflow"):
        _run(client._post_with_retries(session, _TEACHER, {"prompt": [[1]]}, {}))

    assert len(session.requests) == 1


# ---------------------------------------------------------------------------
# persistent worker — dummy-value scoring
# ---------------------------------------------------------------------------


def test_score_group_sequential_loops():
    """Two successive asyncio.run calls (different loops) return dummy values."""
    input_ids = torch.tensor([[1, 2, 3]])
    poster = _FakePoster(
        _canned_response([[1, 2, 3]], fill=0.25),
        _canned_response([[1, 2, 3]], fill=0.75),
    )
    client = _client_with(poster)

    out1 = asyncio.run(client.score_group(_TEACHER, input_ids))
    out2 = asyncio.run(client.score_group(_TEACHER, input_ids))

    assert torch.allclose(out1, torch.tensor([[0.0, 0.25, 0.25]]))
    assert torch.allclose(out2, torch.tensor([[0.0, 0.75, 0.75]]))
    _run(client.close())


def test_score_group_concurrent_threads():
    """Concurrent score_group calls from N threads all return dummy values."""
    n_threads = 4
    input_ids = torch.tensor([[1, 2, 3]])
    poster = _FakePoster(*([_canned_response([[1, 2, 3]], fill=0.5)] * n_threads))
    client = _client_with(poster)

    results: list[torch.Tensor] = []

    def _score():
        results.append(asyncio.run(client.score_group(_TEACHER, input_ids)))

    threads = [
        threading.Thread(target=_score, daemon=True) for _ in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(results) == n_threads
    expected = torch.tensor([[0.0, 0.5, 0.5]])
    for out in results:
        assert torch.allclose(out, expected)
    _run(client.close())
