"""Exercise the existing HTTP worker lifecycle and dense scoring contract."""

import asyncio
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading

import pytest
import torch

from nemo_rl.algorithms.x_token.remote_teacher import (
    TeacherCapabilityError,
    XTokenFullLogprobClient,
)


@pytest.fixture
def client_config():
    return {"url": "http://unused", "model": "fixture", "model_vocab_size": 3,
            "api_key_env": None, "max_score_elements": 100,
            "max_response_bytes": 10000, "request_timeout_s": 2,
            "max_attempts": 2, "retry_backoff_s": 0, "normalization_atol": 1e-5}


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


@pytest.mark.parametrize("scenario", ["ok", "retry", "topk", "echo", "duplicate", "limit", "fatal"])
def test_live_http_lifecycle_alignment_and_bounds(client_config, scenario):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            status = 503 if scenario == "retry" and len(requests) == 1 else 200
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
        else:
            with pytest.raises(TeacherCapabilityError):
                asyncio.run(call)
        assert len(requests) == (2 if scenario == "retry" else 1)
        assert requests[0]["prompt"] == [[1, 2, 1], [1, 2, 1]]
        assert requests[0]["prompt_logprobs"] == requests[0]["logprobs"] == -1
    finally:
        asyncio.run(client.close())
        server.shutdown()
        server.server_close()
        thread.join()


def test_preflight_size_guard_makes_no_request(client_config):
    client_config["max_score_elements"] = 1
    client = XTokenFullLogprobClient(client_config)
    try:
        with pytest.raises(ValueError, match="max_score_elements"):
            asyncio.run(client.score_group({}, torch.tensor([[1, 2]]), torch.tensor([2])))
    finally:
        asyncio.run(client.close())
