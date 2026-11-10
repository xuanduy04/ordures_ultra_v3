import asyncio
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from nemo_rl.algorithms.opd_outsource import VLLMTeacherLogprobClient
from nemo_rl.algorithms.x_token.remote_teacher import (
    TeacherCapabilityError,
    RemoteTeacherPolicy,
    XTokenFullLogprobClient,
    XTokenTopKLogprobClient,
)


@pytest.fixture
def client_config():
    return {"url": "http://unused", "model": "fixture", "model_vocab_size": 3,
            "api_key_env": None, "max_score_elements": 100,
            "max_response_bytes": 10000, "request_timeout_s": 2,
            "max_attempts": 2, "retry_backoff_s": 0, "normalization_atol": 1e-5}


def test_shared_lifecycle_keeps_xtoken_request_and_parser_overrides():
    assert XTokenFullLogprobClient.__bases__ == (VLLMTeacherLogprobClient,)
    for method in ("_ensure_worker", "_worker_main", "score_group", "close"):
        assert getattr(XTokenFullLogprobClient, method) is getattr(VLLMTeacherLogprobClient, method)
    assert XTokenFullLogprobClient._score_on_loop is not VLLMTeacherLogprobClient._score_on_loop
    assert XTokenTopKLogprobClient._score_on_loop is not XTokenFullLogprobClient._score_on_loop


def dense_choice():
    return {"index": 0, "prompt_token_ids": [1, 2, 1],
            "prompt_logprobs": [None, {str(i): {"logprob": math.log(p)} for i, p in enumerate([.2, .3, .5])},
                               {str(i): {"logprob": math.log(p)} for i, p in enumerate([.5, .4, .1])}]}


def parser(config):
    result = object.__new__(XTokenFullLogprobClient)
    result.config = config
    return result


def test_numeric_dense_and_temperature_invariance(client_config):
    scores = parser(client_config)._parse_dense_choice(dense_choice(), 3, 3)[1:]
    assert torch.allclose(scores.exp(), torch.tensor([[.2, .3, .5], [.5, .4, .1]]))
    raw = scores + torch.tensor([[9.], [-11.]])
    for temperature in [.7, 1., 2.]:
        torch.testing.assert_close((scores / temperature).softmax(-1), (raw / temperature).softmax(-1))


def test_numeric_placeholder_fallback(client_config):
    choice = dense_choice()
    choice["logprobs"] = {"top_logprobs": [None, *[
        {"token_id:" + key: value["logprob"] for key, value in row.items()}
        for row in choice.pop("prompt_logprobs")[1:]]]}
    assert parser(client_config)._parse_dense_choice(choice, 3, 3).shape == (3, 3)


@pytest.mark.parametrize("mutation", ["topk", "wrong_ids", "nan", "positive_inf", "unnormalized", "no_null", "short"])
def test_reject_incomplete_or_invalid_distribution(client_config, mutation):
    choice = dense_choice()
    row = choice["prompt_logprobs"][1]
    if mutation == "topk":
        del row["2"]
    elif mutation == "wrong_ids":
        row["7"] = row.pop("2")
    elif mutation == "nan":
        row["0"]["logprob"] = float("nan")
    elif mutation == "positive_inf":
        row["0"]["logprob"] = float("inf")
    elif mutation == "unnormalized":
        row["0"]["logprob"] = 0.
    elif mutation == "no_null":
        choice["prompt_logprobs"][0] = row
    elif mutation == "short":
        choice["prompt_logprobs"].pop()
    with pytest.raises(TeacherCapabilityError):
        parser(client_config)._parse_dense_choice(choice, 3, 3)


@pytest.mark.parametrize("scenario", ["ok", "retry", "exhaust", "topk", "echo", "duplicate", "limit", "fatal"])
def test_live_http_lifecycle_alignment_and_bounds(client_config, scenario, monkeypatch):
    requests = []
    authorizations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            authorizations.append(self.headers["Authorization"])
            status = 503 if scenario == "exhaust" or scenario == "retry" and len(requests) == 1 else 200
            if scenario == "fatal":
                status = 400
            choices = [copy.deepcopy(dense_choice()) for _ in payload["prompt"]]
            for index, choice in enumerate(choices):
                choice["index"] = index
            if scenario == "echo":
                choices[0]["prompt_token_ids"] = [0, 0, 0]
            if scenario == "duplicate":
                choices[-1]["index"] = 0
            if scenario == "topk":
                del choices[0]["prompt_logprobs"][1]["2"]
            body = json.dumps({"choices": list(reversed(choices))}).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client_config["url"] = "http://127.0.0.1:" + str(server.server_port)
    client_config["api_key_env"] = "XTOKEN_TEST_TEACHER_KEY"
    monkeypatch.setenv("XTOKEN_TEST_TEACHER_KEY", "fixture-only-key")
    if scenario == "limit":
        client_config["max_response_bytes"] = 10
    client = XTokenFullLogprobClient(client_config)
    teacher = {"url": client_config["url"], "model": "fixture"}
    try:
        call = client.score_group(teacher, torch.tensor([[1, 2, 1], [1, 2, 1]]), torch.tensor([3, 3]))
        if scenario in {"ok", "retry"}:
            result = asyncio.run(call)
            torch.testing.assert_close(result[0, :2].exp(), torch.tensor([[.2, .3, .5], [.5, .4, .1]]))
            assert result[0, 2].eq(0).all()
        elif scenario == "exhaust":
            with pytest.raises(RuntimeError, match="Transient"):
                asyncio.run(call)
        else:
            with pytest.raises(TeacherCapabilityError):
                asyncio.run(call)
        assert len(requests) == (2 if scenario in {"retry", "exhaust"} else 1)
        assert authorizations == ["Bearer fixture-only-key"] * len(requests)
        assert requests[0]["prompt"] == [[1, 2, 1], [1, 2, 1]]
        assert requests[0]["prompt_logprobs"] == -1
        assert requests[0]["logprobs"] is None
    finally:
        asyncio.run(client.close())
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("client_type", [XTokenFullLogprobClient, XTokenTopKLogprobClient])
@pytest.mark.parametrize("scenario", ["retry", "timeout", "fatal", "declared_limit", "streamed_limit"])
def test_xtoken_bounded_transport_does_not_use_unlimited_opd_retry(client_config, client_type, scenario, monkeypatch):
    client_config["top_k"] = 2
    client_config["max_response_bytes"] = 10
    client = client_type(client_config)
    requests = []
    body = b'{"ok":1}'

    class Response:
        status = 200
        content_length = 11 if scenario == "declared_limit" else None

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            if scenario == "timeout":
                raise asyncio.TimeoutError()
            self.status = 400 if scenario == "fatal" else 503 if scenario == "retry" and len(requests) == 1 else 200
            return self

        async def __aexit__(self, *args):
            return False

        async def iter_chunked(self, size):
            yield body
            if scenario == "streamed_limit":
                yield body

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()

    client._session = SimpleNamespace(post=post)
    legacy_retry = AsyncMock(side_effect=AssertionError("X-Token must retain bounded retry"))
    monkeypatch.setattr(client, "_post_with_retries", legacy_retry)
    try:
        call = client._dense_request({"url": "http://fixture/", "model": "fixture"}, {"prompt": [[1, 2]]}, {})
        if scenario == "retry":
            assert asyncio.run(call) == {"ok": 1}
        elif scenario == "timeout":
            with pytest.raises(asyncio.TimeoutError):
                asyncio.run(call)
        else:
            with pytest.raises(TeacherCapabilityError):
                asyncio.run(call)
        assert len(requests) == (2 if scenario in {"retry", "timeout"} else 1)
        assert all(url == "http://fixture/v1/completions" for url, _ in requests)
        assert all(kwargs["timeout"].total == client_config["request_timeout_s"] for _, kwargs in requests)
        legacy_retry.assert_not_awaited()
    finally:
        asyncio.run(client.close())


def test_preflight_size_guard_makes_no_request(client_config):
    client_config["max_score_elements"] = 1
    client = XTokenFullLogprobClient(client_config)
    try:
        with pytest.raises(ValueError, match="max_score_elements"):
            asyncio.run(client.score_group({}, torch.tensor([[1, 2]]), torch.tensor([2])))
    finally:
        asyncio.run(client.close())


@pytest.mark.parametrize("length", [-1, 1, 4])
def test_preflight_rejects_invalid_lengths(client_config, length):
    client = XTokenFullLogprobClient(client_config)
    try:
        with pytest.raises(ValueError, match="sequence"):
            asyncio.run(client.score_group({}, torch.tensor([[1, 2, 1]]), torch.tensor([length])))
    finally:
        asyncio.run(client.close())


@pytest.mark.parametrize("fits", [True, False])
def test_remote_policy_preallocates_and_bounds_global_batch(client_config, fits):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    client_config["max_score_elements"] = 36 if fits else 35
    policy = RemoteTeacherPolicy(client_config)
    score = AsyncMock(side_effect=[torch.ones(2, 3, 3), torch.full((2, 3, 3), 2.)])
    policy.client = SimpleNamespace(score_group=score)
    data = BatchedDataDict(input_ids=torch.ones(4, 3, dtype=torch.long), input_lengths=torch.full((4,), 3))
    if fits:
        result = policy.get_full_logits(data, micro_batch_size=2)
        torch.testing.assert_close(result[:2], torch.ones(2, 3, 3))
        torch.testing.assert_close(result[2:], torch.full((2, 3, 3), 2.))
        assert score.await_count == 2
    else:
        with pytest.raises(ValueError, match="Global"):
            policy.get_full_logits(data, micro_batch_size=2)
        score.assert_not_awaited()
